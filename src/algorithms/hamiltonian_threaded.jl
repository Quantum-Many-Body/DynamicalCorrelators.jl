# Graph execution of Jordan-MPO effective Hamiltonians.
#
# Static graph structure, per-operator mutable state, and per-worker resources
# have separate owners. Flat operators are independent-node graphs; shared-prefix
# operators add T -> S/BE and S -> AA/AB dependencies. All tensor expressions are
# unchanged from the validated path/staged kernels.

"""
Set the default threaded-Hamiltonian worker cap (0 = fused fallback on new
operators, 1 = serial). `tree` selects shared-prefix graph construction for
subsequently constructed operators. Existing operators keep their graph but
continue to follow the global cap unless constructed with a fixed policy cap.
"""
function set_threaded_hamiltonian!(n::Integer; tree::Union{Bool, Nothing} = nothing)
    n >= 0 || throw(ArgumentError("threaded-hamiltonian cap must be >= 0, got $n"))
    tree === nothing || (_TREE_ENABLED[] = tree)
    return _THREADED_HAMILTONIAN_THREADS[] = Int(n)
end
set_threaded_hamiltonian!(enable::Bool; tree::Union{Bool, Nothing} = nothing) =
    set_threaded_hamiltonian!(enable ? Threads.nthreads() : 0; tree)

threaded_hamiltonian_threads() =
    _THREADED_HAMILTONIAN_THREADS[] < 0 ? Threads.nthreads() :
        _THREADED_HAMILTONIAN_THREADS[]

const _THREADED_HAMILTONIAN_THREADS = Ref{Int}(-1)
const _TREE_ENABLED = Ref{Bool}(true)

struct _ExecutionPolicy
    cap::Union{Nothing, Int}
    tree::Bool
end
_ExecutionPolicy(; cap = nothing, tree = _TREE_ENABLED[]) =
    _ExecutionPolicy(cap, tree)
_policy_cap(policy::_ExecutionPolicy) =
    policy.cap === nothing ? threaded_hamiltonian_threads() : policy.cap

const _EMPTY_META = (; gl = 0, gc = 0, mid = 0)

struct _GraphNode
    kind::Symbol
    weight::Float64
    writes_y::Bool
    apply::Function
    meta::NamedTuple
end

struct _GraphPlan
    mode::Symbol
    nodes::Vector{_GraphNode}
    roots::Vector{Int}
    indegree::Vector{Int}
    dependent_offsets::Vector{Int}
    dependents::Vector{Int}
    topology::Vector{Int}
    slot_producers::Vector{Int}
    slot_user_offsets::Vector{Int}
    slot_users::Vector{Int}
    rows_T::Vector{Int}
    pairs_S::Vector{Tuple{Int, Int}}
    nleft::Int
    nmid::Int
end

mutable struct _GraphBuilder
    mode::Symbol
    nodes::Vector{_GraphNode}
    dependencies::Vector{Vector{Int}}
    reads::Vector{Vector{Int}}
    writes::Vector{Vector{Int}}
    slot_producers::Vector{Int}
end
_GraphBuilder(mode::Symbol) =
    _GraphBuilder(mode, _GraphNode[], Vector{Int}[], Vector{Int}[], Vector{Int}[], Int[])

function _add_slot!(builder::_GraphBuilder)
    push!(builder.slot_producers, 0)
    return length(builder.slot_producers)
end
_add_slots!(builder::_GraphBuilder, n::Int) = [_add_slot!(builder) for _ in 1:n]

function _add_node!(
        builder::_GraphBuilder, kind::Symbol, weight, apply;
        writes_y::Bool = true, deps = Int[], reads = Int[], produces = Int[],
        meta = _EMPTY_META
    )
    push!(builder.nodes, _GraphNode(kind, Float64(weight), writes_y, apply, meta))
    push!(builder.dependencies, collect(Int, deps))
    push!(builder.reads, collect(Int, reads))
    push!(builder.writes, collect(Int, produces))
    return length(builder.nodes)
end

function _csr_from_parents(parents::Vector{Vector{Int}}, n::Int)
    offsets = Vector{Int}(undef, n + 1)
    offsets[1] = 1
    for i in 1:n
        offsets[i + 1] = offsets[i] + length(parents[i])
    end
    values = Vector{Int}(undef, offsets[end] - 1)
    for i in 1:n
        values[offsets[i]:(offsets[i + 1] - 1)] = parents[i]
    end
    return offsets, values
end

function _freeze_graph(
        builder::_GraphBuilder;
        rows_T = Int[], pairs_S = Tuple{Int, Int}[], nleft = 0, nmid = 0
    )
    nodes = builder.nodes
    n = length(nodes)
    ns = length(builder.slot_producers)
    producers = copy(builder.slot_producers)
    dependents = [Int[] for _ in 1:n]
    indegree = zeros(Int, n)
    slot_users = [Int[] for _ in 1:ns]

    for i in 1:n
        deps = builder.dependencies[i]
        all(d -> 1 <= d < i, deps) ||
            error("graph node $i has a forward or invalid dependency: $deps")
        for d in deps
            push!(dependents[d], i)
            indegree[i] += 1
        end
        for s in builder.writes[i]
            1 <= s <= ns || error("graph node $i writes invalid slot $s")
            producers[s] == 0 || error("slot $s has multiple producers")
            producers[s] = i
            push!(slot_users[s], i)
        end
        for s in builder.reads[i]
            1 <= s <= ns || error("graph node $i reads invalid slot $s")
            producer = producers[s]
            producer > 0 || error("slot $s is read before it has a producer")
            producer == i || producer in deps ||
                error("graph node $i reads slot $s without depending on producer $producer")
            push!(slot_users[s], i)
        end
    end
    all(>(0), producers) || error("not every graph slot has a producer")

    roots = findall(==(0), indegree)
    remaining = copy(indegree)
    ready = copy(roots)
    topology = Int[]
    sizehint!(topology, n)
    while !isempty(ready)
        best = argmax(i -> (nodes[ready[i]].weight, -ready[i]), eachindex(ready))
        id = ready[best]
        deleteat!(ready, best)
        push!(topology, id)
        for child in dependents[id]
            remaining[child] -= 1
            remaining[child] == 0 && push!(ready, child)
        end
    end
    length(topology) == n || error("cycle detected in Hamiltonian graph")

    sort!(roots; by = i -> (-nodes[i].weight, i))
    dependent_offsets, dependent_values = _csr_from_parents(dependents, n)
    slot_user_offsets, slot_user_values = _csr_from_parents(slot_users, ns)
    return _GraphPlan(
        builder.mode, nodes, roots, indegree, dependent_offsets, dependent_values,
        topology, producers, slot_user_offsets, slot_user_values,
        collect(Int, rows_T), collect(Tuple{Int, Int}, pairs_S), nleft, nmid
    )
end

mutable struct _MatvecState
    slots::Vector{Any}
    input_space::Any
    input_eltype::Any
    input_valid::Bool
    remaining::Vector{Threads.Atomic{Int}}
    heap::Vector{Int}
    condition::Threads.Condition
    done::Int
    failed::Bool
    trace_enabled::Bool
    trace_worker::Vector{Int}
    trace_start::Vector{UInt64}
    trace_stop::Vector{UInt64}
end

function _MatvecState(plan::_GraphPlan)
    n = length(plan.nodes)
    heap = Int[]
    sizehint!(heap, n)
    return _MatvecState(
        fill(nothing, length(plan.slot_producers)), nothing, nothing, false,
        [Threads.Atomic{Int}(0) for _ in 1:n], heap, Threads.Condition(), 0, false,
        false, zeros(Int, n), zeros(UInt64, n), zeros(UInt64, n)
    )
end

function _ensure_state!(state::_MatvecState, plan::_GraphPlan, x)
    isempty(state.slots) && return state
    valid = state.input_valid
    if !valid || state.input_space != space(x) || state.input_eltype != eltype(x)
        fill!(state.slots, nothing)
        state.input_space = space(x)
        state.input_eltype = eltype(x)
        state.input_valid = true
    end
    return state
end

function _release_state!(state::_MatvecState)
    fill!(state.slots, nothing)
    state.input_space = nothing
    state.input_eltype = nothing
    state.input_valid = false
    empty!(state.heap)
    state.done = 0
    state.failed = false
    state.trace_enabled = false
    return state
end

mutable struct _WorkerResources
    output_space::Any
    output_eltype::Any
    output_valid::Bool
    buffers::Vector{Any}
    scratch::Vector{BufferAllocator}
    tmerge::Vector{Float64}
    nrun::Vector{Int}
    wrote::Vector{Bool}
    wait_ns::Vector{UInt64}
    compute_ns::Vector{UInt64}
    active_workers::Int
    matvec_lock::ReentrantLock
    merge_lock::ReentrantLock
end

function _WorkerResources()
    return _WorkerResources(
        nothing, nothing, false, Any[], BufferAllocator[], Float64[], Int[], Bool[],
        UInt64[], UInt64[], 0, ReentrantLock(), ReentrantLock()
    )
end

function _ensure_workers!(ws::_WorkerResources, y, n::Int, accumulators::Bool)
    if !ws.output_valid || ws.output_space != space(y) || ws.output_eltype != eltype(y)
        empty!(ws.buffers)
        ws.output_space = space(y)
        ws.output_eltype = eltype(y)
        ws.output_valid = true
    end
    if accumulators
        while length(ws.buffers) < n
            push!(ws.buffers, vi_zerovector(y, eltype(y)))
        end
    end
    while length(ws.scratch) < n
        push!(ws.scratch, BufferAllocator())
    end
    for field in (:tmerge, :nrun, :wrote, :wait_ns, :compute_ns)
        values = getfield(ws, field)
        while length(values) < n
            push!(values, zero(eltype(values)))
        end
    end
    fill!(ws.tmerge, 0.0)
    fill!(ws.nrun, 0)
    fill!(ws.wrote, false)
    fill!(ws.wait_ns, 0)
    fill!(ws.compute_ns, 0)
    ws.active_workers = n
    return ws
end

function _release_workers!(ws::_WorkerResources)
    empty!(ws.buffers)
    empty!(ws.scratch)
    empty!(ws.tmerge)
    empty!(ws.nrun)
    empty!(ws.wrote)
    empty!(ws.wait_ns)
    empty!(ws.compute_ns)
    ws.output_space = nothing
    ws.output_eltype = nothing
    ws.output_valid = false
    ws.active_workers = 0
    return ws
end

@inline function _apply_node!(node::_GraphNode, y, x, scratch, first, state::_MatvecState)
    node.apply(y, x, scratch, first, state)
    return nothing
end

function _run_node!(
        id::Int, plan::_GraphPlan, state::_MatvecState, ws::_WorkerResources,
        w::Int, y, x, scratch, first
    )
    node = plan.nodes[id]
    traced = state.trace_enabled
    start = traced ? time_ns() : UInt64(0)
    _apply_node!(node, y, x, scratch, first, state)
    stop = traced ? time_ns() : UInt64(0)
    if traced
        state.trace_worker[id] = w
        state.trace_start[id] = start
        state.trace_stop[id] = stop
        ws.compute_ns[w] += stop - start
    end
    ws.nrun[w] += 1
    node.writes_y && (ws.wrote[w] = true)
    return nothing
end

function _reset_trace!(state::_MatvecState)
    state.trace_enabled || return state
    fill!(state.trace_worker, 0)
    fill!(state.trace_start, 0)
    fill!(state.trace_stop, 0)
    return state
end

@inline _higher_priority(nodes, a::Int, b::Int) =
    nodes[a].weight > nodes[b].weight ||
    (nodes[a].weight == nodes[b].weight && a < b)

function _heap_push!(heap::Vector{Int}, id::Int, nodes)
    push!(heap, id)
    i = length(heap)
    while i > 1
        parent = i >>> 1
        _higher_priority(nodes, heap[i], heap[parent]) || break
        heap[i], heap[parent] = heap[parent], heap[i]
        i = parent
    end
    return heap
end

function _heap_pop!(heap::Vector{Int}, nodes)
    root = heap[1]
    last = pop!(heap)
    if !isempty(heap)
        heap[1] = last
        i = 1
        n = length(heap)
        while true
            left = 2i
            left > n && break
            right = left + 1
            child = right <= n && _higher_priority(nodes, heap[right], heap[left]) ? right : left
            _higher_priority(nodes, heap[child], heap[i]) || break
            heap[i], heap[child] = heap[child], heap[i]
            i = child
        end
    end
    return root
end

function _execute_serial!(y, plan::_GraphPlan, state::_MatvecState, ws::_WorkerResources, x)
    _ensure_workers!(ws, y, 1, false)
    scratch = ws.scratch[1]
    for id in plan.topology
        _run_node!(id, plan, state, ws, 1, y, x, scratch, false)
    end
    return y
end

function _execute_independent!(y, plan::_GraphPlan, state::_MatvecState, ws::_WorkerResources, x)
    order = plan.topology
    total = length(order)
    nt = min(ws.active_workers, total)
    idx = Threads.Atomic{Int}(1)
    Threads.@sync for w in 1:nt
        Threads.@spawn begin
            worker_y = ws.buffers[w]
            worker_buf = ws.scratch[w]
            while true
                next_index = Threads.atomic_add!(idx, 1)
                next_index > total && break
                node_id = order[next_index]
                _run_node!(node_id, plan, state, ws, w,
                    worker_y, x, worker_buf, !ws.wrote[w])
            end
            if ws.wrote[w]
                merge_start = time_ns()
                lock(ws.merge_lock) do
                    add!(y, worker_y, One(), One())
                end
                ws.tmerge[w] = (time_ns() - merge_start) / 1e9
            end
        end
    end
    return y
end

function _dag_next!(state::_MatvecState, plan::_GraphPlan)
    lock(state.condition)
    try
        while isempty(state.heap) && !state.failed && state.done < length(plan.nodes)
            wait(state.condition)
        end
        if state.failed || state.done >= length(plan.nodes)
            return 0
        end
        return _heap_pop!(state.heap, plan.nodes)
    finally
        unlock(state.condition)
    end
end

function _dag_publish!(id::Int, state::_MatvecState, plan::_GraphPlan)
    for j in plan.dependent_offsets[id]:(plan.dependent_offsets[id + 1] - 1)
        child = plan.dependents[j]
        if Threads.atomic_sub!(state.remaining[child], 1) == 1
            lock(state.condition) do
                _heap_push!(state.heap, child, plan.nodes)
                notify(state.condition; all = false)
            end
        end
    end
    lock(state.condition) do
        state.done += 1
        state.done == length(plan.nodes) && notify(state.condition; all = true)
    end
    return nothing
end

function _execute_dag!(y, plan::_GraphPlan, state::_MatvecState, ws::_WorkerResources, x)
    n = length(plan.nodes)
    for i in 1:n
        Threads.atomic_xchg!(state.remaining[i], plan.indegree[i])
    end
    lock(state.condition) do
        empty!(state.heap)
        for root in plan.roots
            _heap_push!(state.heap, root, plan.nodes)
        end
        state.done = 0
        state.failed = false
    end
    nt = ws.active_workers
    Threads.@sync for w in 1:nt
        Threads.@spawn begin
            worker_y = ws.buffers[w]
            worker_buf = ws.scratch[w]
            try
                while true
                    wait_start = time_ns()
                    next_node = _dag_next!(state, plan)
                    ws.wait_ns[w] += time_ns() - wait_start
                    next_node == 0 && break
                    _run_node!(next_node, plan, state, ws, w,
                        worker_y, x, worker_buf, !ws.wrote[w])
                    _dag_publish!(next_node, state, plan)
                end
            catch
                lock(state.condition) do
                    state.failed = true
                    notify(state.condition; all = true)
                end
                rethrow()
            end
        end
    end
    for w in 1:nt
        ws.wrote[w] || continue
        merge_start = time_ns()
        add!(y, ws.buffers[w], One(), One())
        ws.tmerge[w] = (time_ns() - merge_start) / 1e9
    end
    return y
end

function _apply_plan!(y, plan::_GraphPlan, state::_MatvecState,
        ws::_WorkerResources, policy::_ExecutionPolicy, x)
    lock(ws.matvec_lock) do
        _ensure_state!(state, plan, x)
        total = length(plan.nodes)
        total == 0 && return y
        nt = clamp(min(_policy_cap(policy), Threads.nthreads()), 1, total)
        _reset_trace!(state)
        if nt == 1
            return _execute_serial!(y, plan, state, ws, x)
        end
        _ensure_workers!(ws, y, nt, true)
        if isempty(plan.dependents)
            return _execute_independent!(y, plan, state, ws, x)
        end
        return _execute_dag!(y, plan, state, ws, x)
    end
end

# Persistent intermediate storage lives in MatvecState. Values are recomputed on
# every matvec; only tensor storage is reused.
struct _StageCache{T<:Tuple}
    tensors::T
end
_stage_cache(tensors::T) where {T<:Tuple} = _StageCache{T}(tensors)

@inline function _getslot!(make, state::_MatvecState, id::Int)
    value = state.slots[id]
    if value === nothing
        value = make()
        state.slots[id] = value
    end
    return value
end

_task_cost(noperators::Int, nidentity::Int, tensors...) =
    sum(numind, tensors; init = 0) - 2 * (noperators - nidentity)

_isidentity_A(W, I) =
    haskey(W.scalars, CartesianIndex(I[1] + 1, 1, 1, I[4] + 1))
_isidentity_B(W, I) =
    haskey(W.scalars, CartesianIndex(I[1] + 1, 1, 1, size(W, 4)))
_isidentity_C(W, I) =
    haskey(W.scalars, CartesianIndex(1, 1, 1, I[3] + 1))

function _apply_onsite!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3] = x[-1 1; -3] * op[-2; 1]
    else
        @plansor allocator = buffer y[-1 -2; -3] += x[-1 1; -3] * op[-2; 1]
    end
    return nothing
end
function _apply_finished!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3] = op[-1; 1] * x[1 -2; -3]
    else
        @plansor allocator = buffer y[-1 -2; -3] += op[-1; 1] * x[1 -2; -3]
    end
    return nothing
end
function _apply_notstarted!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3] = x[-1 -2; 1] * op[1; -3]
    else
        @plansor allocator = buffer y[-1 -2; -3] += x[-1 -2; 1] * op[1; -3]
    end
    return nothing
end
function _apply_II!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] = x[-1 -2; 1 -4] * op[-3; 1]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] += x[-1 -2; 1 -4] * op[-3; 1]
    end
    return nothing
end
function _apply_ID!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] = x[-1 -2; -3 1] * op[-4; 1]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] += x[-1 -2; -3 1] * op[-4; 1]
    end
    return nothing
end
function _apply_DE!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] = x[-1 1; -3 -4] * op[-2; 1]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] += x[-1 1; -3 -4] * op[-2; 1]
    end
    return nothing
end
function _apply_EE!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] = x[1 -2; -3 -4] * op[-1; 1]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] += x[1 -2; -3 -4] * op[-1; 1]
    end
    return nothing
end

function _a_cache(x, GL, A)
    t1 = @plansor allocator = DefaultAllocator() t1[-1 -2; -3 -4] :=
        GL[-1 -2; 1] * x[1 -4; -3]
    t2 = @plansor allocator = DefaultAllocator() t2[-1 -2; -3 -4] :=
        t1[-1 2; -3 1] * A[2 -2; 1 -4]
    return _stage_cache((t1, t2))
end
function _apply_A_cached!(y, state, slot, buffer, GL, x, A, GR, first::Bool)
    c = _getslot!(state, slot) do
        _a_cache(x, GL, A)
    end
    t1, t2 = c.tensors
    @plansor allocator = buffer t1[-1 -2; -3 -4] =
        GL[-1 -2; 1] * x[1 -4; -3]
    @plansor allocator = buffer t2[-1 -2; -3 -4] =
        t1[-1 2; -3 1] * A[2 -2; 1 -4]
    if first
        @plansor allocator = buffer y[-1 -2; -3] =
            t2[-1 -2; 1 2] * GR[1 2; -3]
    else
        @plansor allocator = buffer y[-1 -2; -3] +=
            t2[-1 -2; 1 2] * GR[1 2; -3]
    end
    return nothing
end

function _b_cache(x, GL)
    t = @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4] :=
        GL[-1 -2; 1] * x[1 -4; -3]
    return _stage_cache((t,))
end
function _apply_B_cached!(y, state, slot, buffer, GL, x, B, first::Bool)
    c = _getslot!(state, slot) do
        _b_cache(x, GL)
    end
    t = c.tensors[1]
    @plansor allocator = buffer t[-1 -2; -3 -4] =
        GL[-1 -2; 1] * x[1 -4; -3]
    if first
        @plansor allocator = buffer y[-1 -2; -3] =
            t[-1 2; -3 1] * B[2 -2; 1]
    else
        @plansor allocator = buffer y[-1 -2; -3] +=
            t[-1 2; -3 1] * B[2 -2; 1]
    end
    return nothing
end

function _c_cache(x, C)
    t = @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4] :=
        x[-1 1; -3] * C[-2; 1 -4]
    return _stage_cache((t,))
end
function _apply_C_cached!(y, state, slot, buffer, x, C, GR, first::Bool)
    c = _getslot!(state, slot) do
        _c_cache(x, C)
    end
    t = c.tensors[1]
    @plansor allocator = buffer t[-1 -2; -3 -4] =
        x[-1 1; -3] * C[-2; 1 -4]
    if first
        @plansor allocator = buffer y[-1 -2; -3] =
            t[-1 -2; 1 2] * GR[1 2; -3]
    else
        @plansor allocator = buffer y[-1 -2; -3] +=
            t[-1 -2; 1 2] * GR[1 2; -3]
    end
    return nothing
end

function _cb_cache(x, C1)
    t = @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4 -5] :=
        x[-1 1; -3 -4] * C1[-2; 1 -5]
    return _stage_cache((t,))
end
function _apply_CB_cached!(y, state, slot, buffer, x, C1, B2, first::Bool)
    c = _getslot!(state, slot) do
        _cb_cache(x, C1)
    end
    t = c.tensors[1]
    @plansor allocator = buffer t[-1 -2; -3 -4 -5] =
        x[-1 1; -3 -4] * C1[-2; 1 -5]
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] =
            t[-1 -2; -3 1 2] * B2[2 -4; 1]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] +=
            t[-1 -2; -3 1 2] * B2[2 -4; 1]
    end
    return nothing
end

function _ic_cache(x, C2)
    t = @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4 -5] :=
        x[-1 -2; -3 1] * C2[-5; 1 -4]
    return _stage_cache((t,))
end
function _apply_IC_cached!(y, state, slot, buffer, x, C2, GR, first::Bool)
    c = _getslot!(state, slot) do
        _ic_cache(x, C2)
    end
    t = c.tensors[1]
    @plansor allocator = buffer t[-1 -2; -3 -4 -5] =
        x[-1 -2; -3 1] * C2[-5; 1 -4]
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] =
            t[-1 -2; 1 2 -4] * GR[1 2; -3]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] +=
            t[-1 -2; 1 2 -4] * GR[1 2; -3]
    end
    return nothing
end

function _be_cache(x, GL)
    t = @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4 -5] :=
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    return _stage_cache((t,))
end
function _apply_BE_cached!(y, state, slot, buffer, GL, x, B1, first::Bool)
    c = _getslot!(state, slot) do
        _be_cache(x, GL)
    end
    t = c.tensors[1]
    @plansor allocator = buffer t[-1 -2; -3 -4 -5] =
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] =
            t[-1 2; -3 -4 1] * B1[2 -2; 1]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] +=
            t[-1 2; -3 -4 1] * B1[2 -2; 1]
    end
    return nothing
end

function _aa_cache(x, GL, A1, A2)
    t1 = @plansor allocator = DefaultAllocator() t1[-1 -2; -3 -4 -5] :=
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    t2 = @plansor allocator = DefaultAllocator() t2[-1 -2; -3 -4 -5] :=
        t1[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
    t3 = @plansor allocator = DefaultAllocator() t3[-1 -2; -3 -4 -5] :=
        t2[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
    return _stage_cache((t1, t2, t3))
end
function _apply_AA_cached!(y, state, slot, buffer, GL, x, A1, A2, GR, first::Bool)
    c = _getslot!(state, slot) do
        _aa_cache(x, GL, A1, A2)
    end
    t1, t2, t3 = c.tensors
    @plansor allocator = buffer t1[-1 -2; -3 -4 -5] =
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    @plansor allocator = buffer t2[-1 -2; -3 -4 -5] =
        t1[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
    @plansor allocator = buffer t3[-1 -2; -3 -4 -5] =
        t2[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] =
            t3[-1 -2; 1 2 -4] * GR[1 2; -3]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] +=
            t3[-1 -2; 1 2 -4] * GR[1 2; -3]
    end
    return nothing
end

function _ca_cache(x, C1, A2)
    t1 = @plansor allocator = DefaultAllocator() t1[-1 -2; -3 -4 -5] :=
        x[-1 1; -3 -4] * C1[-2; 1 -5]
    t2 = @plansor allocator = DefaultAllocator() t2[-1 -2; -3 -4 -5] :=
        t1[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
    return _stage_cache((t1, t2))
end
function _apply_CA_cached!(y, state, slot, buffer, x, C1, A2, GR, first::Bool)
    c = _getslot!(state, slot) do
        _ca_cache(x, C1, A2)
    end
    t1, t2 = c.tensors
    @plansor allocator = buffer t1[-1 -2; -3 -4 -5] =
        x[-1 1; -3 -4] * C1[-2; 1 -5]
    @plansor allocator = buffer t2[-1 -2; -3 -4 -5] =
        t1[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] =
            t2[-1 -2; 1 2 -4] * GR[1 2; -3]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] +=
            t2[-1 -2; 1 2 -4] * GR[1 2; -3]
    end
    return nothing
end

function _ab_cache(x, GL, A1)
    s1 = @plansor allocator = DefaultAllocator() s1[-1 -2; -3 -4 -5] :=
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    s2 = @plansor allocator = DefaultAllocator() s2[-1 -2; -3 -4 -5] :=
        s1[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
    return _stage_cache((s1, s2))
end
function _apply_AB_cached!(y, state, slot, buffer, GL, x, A1, B2, first::Bool)
    c = _getslot!(state, slot) do
        _ab_cache(x, GL, A1)
    end
    s1, s2 = c.tensors
    @plansor allocator = buffer s1[-1 -2; -3 -4 -5] =
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    @plansor allocator = buffer s2[-1 -2; -3 -4 -5] =
        s1[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] =
            s2[-1 -2; -3 2 1] * B2[1 -4; 2]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] +=
            s2[-1 -2; -3 2 1] * B2[1 -4; 2]
    end
    return nothing
end

_a_task(GL, A, GR, slot) =
    (y, x, buf, first, state) -> _apply_A_cached!(y, state, slot, buf, GL, x, A, GR, first)
_b_task(GL, B, slot) =
    (y, x, buf, first, state) -> _apply_B_cached!(y, state, slot, buf, GL, x, B, first)
_c_task(C, GR, slot) =
    (y, x, buf, first, state) -> _apply_C_cached!(y, state, slot, buf, x, C, GR, first)
_cb_task(C1, B2, slot) =
    (y, x, buf, first, state) -> _apply_CB_cached!(y, state, slot, buf, x, C1, B2, first)
_ic_task(C2, GR, slot) =
    (y, x, buf, first, state) -> _apply_IC_cached!(y, state, slot, buf, x, C2, GR, first)
_be_task(GL, B1, slot) =
    (y, x, buf, first, state) -> _apply_BE_cached!(y, state, slot, buf, GL, x, B1, first)
_aa_task(GL, A1, A2, GR, slot) =
    (y, x, buf, first, state) -> _apply_AA_cached!(y, state, slot, buf, GL, x, A1, A2, GR, first)
_ca_task(C1, A2, GR, slot) =
    (y, x, buf, first, state) -> _apply_CA_cached!(y, state, slot, buf, x, C1, A2, GR, first)
_ab_task(GL, A1, B2, slot) =
    (y, x, buf, first, state) -> _apply_AB_cached!(y, state, slot, buf, GL, x, A1, B2, first)

function _t_task(GL, slot::Int)
    return (y, x, buf, first, state) -> begin
        T = _getslot!(state, slot) do
            @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4 -5] :=
                GL[-1 -2; 1] * x[1 -5; -3 -4]
        end
        @plansor allocator = buf T[-1 -2; -3 -4 -5] =
            GL[-1 -2; 1] * x[1 -5; -3 -4]
        return nothing
    end
end
function _s_task(A1, tslot::Int, sslot::Int)
    return (y, x, buf, first, state) -> begin
        T = state.slots[tslot]
        S = _getslot!(state, sslot) do
            @plansor allocator = DefaultAllocator() s[-1 -2; -3 -4 -5] :=
                T[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
        end
        @plansor allocator = buf S[-1 -2; -3 -4 -5] =
            T[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
        return nothing
    end
end
function _aa_tail_task(sslot::Int, t3slot::Int, A2, GR)
    return (y, x, buf, first, state) -> begin
        S = state.slots[sslot]
        t3 = _getslot!(state, t3slot) do
            @plansor allocator = DefaultAllocator() t3[-1 -2; -3 -4 -5] :=
                S[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
        end
        @plansor allocator = buf t3[-1 -2; -3 -4 -5] =
            S[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
        if first
            @plansor allocator = buf y[-1 -2; -3 -4] =
                t3[-1 -2; 1 2 -4] * GR[1 2; -3]
        else
            @plansor allocator = buf y[-1 -2; -3 -4] +=
                t3[-1 -2; 1 2 -4] * GR[1 2; -3]
        end
        return nothing
    end
end
function _ab_tail_task(sslot::Int, B2)
    return (y, x, buf, first, state) -> begin
        S = state.slots[sslot]
        if first
            @plansor allocator = buf y[-1 -2; -3 -4] =
                S[-1 -2; -3 2 1] * B2[1 -4; 2]
        else
            @plansor allocator = buf y[-1 -2; -3 -4] +=
                S[-1 -2; -3 2 1] * B2[1 -4; 2]
        end
        return nothing
    end
end
function _be_tail_task(tslot::Int, B1)
    return (y, x, buf, first, state) -> begin
        T = state.slots[tslot]
        if first
            @plansor allocator = buf y[-1 -2; -3 -4] =
                T[-1 2; -3 -4 1] * B1[2 -2; 1]
        else
            @plansor allocator = buf y[-1 -2; -3 -4] +=
                T[-1 2; -3 -4 1] * B1[2 -2; 1]
        end
        return nothing
    end
end

function _t_task_AC(GL, slot::Int)
    return (y, x, buf, first, state) -> begin
        T = _getslot!(state, slot) do
            @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4] :=
                GL[-1 -2; 1] * x[1 -4; -3]
        end
        @plansor allocator = buf T[-1 -2; -3 -4] =
            GL[-1 -2; 1] * x[1 -4; -3]
        return nothing
    end
end
function _a_tail_task_AC(tslot::Int, t2slot::Int, A, GR)
    return (y, x, buf, first, state) -> begin
        T = state.slots[tslot]
        t2 = _getslot!(state, t2slot) do
            @plansor allocator = DefaultAllocator() t2[-1 -2; -3 -4] :=
                T[-1 2; -3 1] * A[2 -2; 1 -4]
        end
        @plansor allocator = buf t2[-1 -2; -3 -4] =
            T[-1 2; -3 1] * A[2 -2; 1 -4]
        if first
            @plansor allocator = buf y[-1 -2; -3] =
                t2[-1 -2; 1 2] * GR[1 2; -3]
        else
            @plansor allocator = buf y[-1 -2; -3] +=
                t2[-1 -2; 1 2] * GR[1 2; -3]
        end
        return nothing
    end
end
function _b_tail_task_AC(tslot::Int, B)
    return (y, x, buf, first, state) -> begin
        T = state.slots[tslot]
        if first
            @plansor allocator = buf y[-1 -2; -3] =
                T[-1 2; -3 1] * B[2 -2; 1]
        else
            @plansor allocator = buf y[-1 -2; -3] +=
                T[-1 2; -3 1] * B[2 -2; 1]
        end
        return nothing
    end
end

function _add_boundary!(builder, Tel, kind::Symbol, apply, op,
        noperators::Int = 0, nidentity::Int = 0)
    ismissing(op) && return Tel
    cost = _task_cost(noperators, nidentity, op)
    _add_node!(builder, kind, cost,
        (y, x, buf, first, state) -> apply(y, op, x, buf, first))
    return promote_type(Tel, eltype(op))
end

function _flat_graph_AC(GL::MPSTensor, W::JordanMPOTensor, GR::MPSTensor)
    builder = _GraphBuilder(:flat)
    GL2, GR2 = GL[2:(end - 1)], GR[2:(end - 1)]
    Tel = Union{}
    for (I, Ab) in nonzero_pairs(W.A)
        GLb, GRb = GL2[I[1]], GR2[I[4]]
        cost = _task_cost(1, Int(_isidentity_A(W, I)), GLb, Ab, GRb)
        slot = _add_slot!(builder)
        _add_node!(builder, :A, cost, _a_task(GLb, Ab, GRb, slot);
            produces = [slot], meta = (; gl = I[1], gc = 0, mid = 0))
        Tel = promote_type(Tel, eltype(GLb), eltype(Ab), eltype(GRb))
    end
    for (I, Bb) in nonzero_pairs(W.B)
        GLb = GL2[I[1]]
        cost = _task_cost(1, Int(_isidentity_B(W, I)), GLb, Bb)
        slot = _add_slot!(builder)
        _add_node!(builder, :B, cost, _b_task(GLb, Bb, slot);
            produces = [slot], meta = (; gl = I[1], gc = 0, mid = 0))
        Tel = promote_type(Tel, eltype(GLb), eltype(Bb))
    end
    for (I, Cb) in nonzero_pairs(W.C)
        GRb = GR2[I[3]]
        cost = _task_cost(1, Int(_isidentity_C(W, I)), Cb, GRb)
        slot = _add_slot!(builder)
        _add_node!(builder, :C, cost, _c_task(Cb, GRb, slot);
            produces = [slot], meta = (; gl = 0, gc = I[3], mid = 0))
        Tel = promote_type(Tel, eltype(Cb), eltype(GRb))
    end
    Tel = _add_boundary!(builder, Tel, :onsite, _apply_onsite!,
        nonzero_length(W.D) > 0 ? only(W.D) : missing)
    Tel = _add_boundary!(builder, Tel, :finished, _apply_finished!,
        size(W, 1) == 1 ? missing : removeunit(GL[end], 2))
    Tel = _add_boundary!(builder, Tel, :notstarted, _apply_notstarted!,
        size(W, 4) == 1 ? missing : removeunit(GR[1], 2))
    return _freeze_graph(builder), Tel
end

function _flat_graph_AC2(GL::MPSTensor, W1::JordanMPOTensor, W2::JordanMPOTensor, GR::MPSTensor)
    builder = _GraphBuilder(:flat)
    GL2, GR2 = GL[2:(end - 1)], GR[2:(end - 1)]
    Tel = Union{}
    for (I1, A1b) in nonzero_pairs(W1.A), (I2, A2b) in nonzero_pairs(W2.A)
        I1[4] == I2[1] || continue
        GLb, GRb = GL2[I1[1]], GR2[I2[4]]
        nid = Int(_isidentity_A(W1, I1)) + Int(_isidentity_A(W2, I2))
        cost = _task_cost(2, nid, GLb, A1b, A2b, GRb)
        slot = _add_slot!(builder)
        _add_node!(builder, :AA, cost, _aa_task(GLb, A1b, A2b, GRb, slot);
            produces = [slot], meta = (; gl = I1[1], gc = 0, mid = I1[4]))
        Tel = promote_type(Tel, eltype(GLb), eltype(A1b), eltype(A2b), eltype(GRb))
    end
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ia, A2b) in nonzero_pairs(W2.A)
        Ic[3] == Ia[1] || continue
        GRb = GR2[Ia[4]]
        nid = Int(_isidentity_C(W1, Ic)) + Int(_isidentity_A(W2, Ia))
        cost = _task_cost(2, nid, C1b, A2b, GRb)
        slot = _add_slot!(builder)
        _add_node!(builder, :CA, cost, _ca_task(C1b, A2b, GRb, slot);
            produces = [slot], meta = (; gl = 0, gc = Ic[3], mid = Ia[1]))
        Tel = promote_type(Tel, eltype(C1b), eltype(A2b), eltype(GRb))
    end
    for (Ia, A1b) in nonzero_pairs(W1.A), (Ib, B2b) in nonzero_pairs(W2.B)
        Ia[4] == Ib[1] || continue
        GLb = GL2[Ia[1]]
        nid = Int(_isidentity_A(W1, Ia)) + Int(_isidentity_B(W2, Ib))
        cost = _task_cost(2, nid, GLb, A1b, B2b)
        slot = _add_slot!(builder)
        _add_node!(builder, :AB, cost, _ab_task(GLb, A1b, B2b, slot);
            produces = [slot], meta = (; gl = Ia[1], gc = 0, mid = Ia[4]))
        Tel = promote_type(Tel, eltype(GLb), eltype(A1b), eltype(B2b))
    end
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ib, B2b) in nonzero_pairs(W2.B)
        Ic[3] == Ib[1] || continue
        nid = Int(_isidentity_C(W1, Ic)) + Int(_isidentity_B(W2, Ib))
        cost = _task_cost(2, nid, C1b, B2b)
        slot = _add_slot!(builder)
        _add_node!(builder, :CB, cost, _cb_task(C1b, B2b, slot);
            produces = [slot], meta = (; gl = 0, gc = Ic[3], mid = Ib[1]))
        Tel = promote_type(Tel, eltype(C1b), eltype(B2b))
    end
    for (Ic, C2b) in nonzero_pairs(W2.C)
        GRb = GR2[Ic[3]]
        cost = _task_cost(1, Int(_isidentity_C(W2, Ic)), C2b, GRb)
        slot = _add_slot!(builder)
        _add_node!(builder, :IC, cost, _ic_task(C2b, GRb, slot);
            produces = [slot], meta = (; gl = 0, gc = Ic[3], mid = 0))
        Tel = promote_type(Tel, eltype(C2b), eltype(GRb))
    end
    for (Ib, B1b) in nonzero_pairs(W1.B)
        GLb = GL2[Ib[1]]
        cost = _task_cost(1, Int(_isidentity_B(W1, Ib)), GLb, B1b)
        slot = _add_slot!(builder)
        _add_node!(builder, :BE, cost, _be_task(GLb, B1b, slot);
            produces = [slot], meta = (; gl = Ib[1], gc = 0, mid = 0))
        Tel = promote_type(Tel, eltype(GLb), eltype(B1b))
    end
    Tel = _add_boundary!(builder, Tel, :II, _apply_II!,
        size(W2, 4) == 1 ? missing : transpose(removeunit(GR[1], 2)))
    Tel = _add_boundary!(builder, Tel, :ID, _apply_ID!,
        nonzero_length(W2.D) > 0 ? only(W2.D) : missing)
    Tel = _add_boundary!(builder, Tel, :DE, _apply_DE!,
        nonzero_length(W1.D) > 0 ? only(W1.D) : missing)
    Tel = _add_boundary!(builder, Tel, :EE, _apply_EE!,
        size(W1, 1) == 1 ? missing : removeunit(GL[end], 2))
    return _freeze_graph(builder), Tel
end

function _tree_graph_AC(GL::MPSTensor, W::JordanMPOTensor, GR::MPSTensor)
    GL2, GR2 = GL[2:(end - 1)], GR[2:(end - 1)]
    aa = Tuple{Int, Any, Any, Int}[]
    be = Tuple{Int, Any, Int}[]
    Tel = Union{}
    for (I, Ab) in nonzero_pairs(W.A)
        a, b = I[1], I[4]
        GLb, GRb = GL2[a], GR2[b]
        push!(aa, (a, Ab, GRb, _task_cost(1, Int(_isidentity_A(W, I)), GLb, Ab, GRb)))
        Tel = promote_type(Tel, eltype(GLb), eltype(Ab), eltype(GRb))
    end
    for (I, Bb) in nonzero_pairs(W.B)
        a = I[1]
        GLb = GL2[a]
        push!(be, (a, Bb, _task_cost(1, Int(_isidentity_B(W, I)), GLb, Bb)))
        Tel = promote_type(Tel, eltype(GLb), eltype(Bb))
    end
    nleft = length(aa) + length(be)
    rows = sort!(unique!(vcat(Int[t[1] for t in aa], Int[t[1] for t in be])))
    (nleft == 0 || nleft == length(rows)) && return _flat_graph_AC(GL, W, GR)

    builder = _GraphBuilder(:tree)
    Tslots = Dict(a => _add_slot!(builder) for a in rows)
    Tnodes = Dict{Int, Int}()
    for a in rows
        Tnodes[a] = _add_node!(builder, :T, numind(GL2[a]), _t_task_AC(GL2[a], Tslots[a]);
            writes_y = false, produces = [Tslots[a]], meta = (; gl = a, gc = 0, mid = 0))
    end
    for (I, Cb) in nonzero_pairs(W.C)
        GRb = GR2[I[3]]
        cost = _task_cost(1, Int(_isidentity_C(W, I)), Cb, GRb)
        slot = _add_slot!(builder)
        _add_node!(builder, :C, cost, _c_task(Cb, GRb, slot);
            produces = [slot], meta = (; gl = 0, gc = I[3], mid = 0))
        Tel = promote_type(Tel, eltype(Cb), eltype(GRb))
    end
    Tel = _add_boundary!(builder, Tel, :onsite, _apply_onsite!,
        nonzero_length(W.D) > 0 ? only(W.D) : missing)
    Tel = _add_boundary!(builder, Tel, :finished, _apply_finished!,
        size(W, 1) == 1 ? missing : removeunit(GL[end], 2))
    Tel = _add_boundary!(builder, Tel, :notstarted, _apply_notstarted!,
        size(W, 4) == 1 ? missing : removeunit(GR[1], 2))
    for (a, Ab, GRb, cost) in aa
        slot = _add_slot!(builder)
        _add_node!(builder, :A, cost, _a_tail_task_AC(Tslots[a], slot, Ab, GRb);
            deps = [Tnodes[a]], reads = [Tslots[a]], produces = [slot],
            meta = (; gl = a, gc = 0, mid = 0))
    end
    for (a, Bb, cost) in be
        _add_node!(builder, :B, cost, _b_tail_task_AC(Tslots[a], Bb);
            deps = [Tnodes[a]], reads = [Tslots[a]], meta = (; gl = a, gc = 0, mid = 0))
    end
    return _freeze_graph(builder; rows_T = rows, nleft = nleft), Tel
end

function _tree_graph_AC2(GL::MPSTensor, W1::JordanMPOTensor, W2::JordanMPOTensor, GR::MPSTensor)
    GL2, GR2 = GL[2:(end - 1)], GR[2:(end - 1)]
    aa = Tuple{Int, Int, Any, Any, Any, Int}[]
    ab = Tuple{Int, Int, Any, Any, Int}[]
    be = Tuple{Int, Any, Int}[]
    Tel = Union{}
    for (I1, A1b) in nonzero_pairs(W1.A), (I2, A2b) in nonzero_pairs(W2.A)
        I1[4] == I2[1] || continue
        a, m, b = I1[1], I1[4], I2[4]
        GLb, GRb = GL2[a], GR2[b]
        nid = Int(_isidentity_A(W1, I1)) + Int(_isidentity_A(W2, I2))
        push!(aa, (a, m, A1b, A2b, GRb, _task_cost(2, nid, GLb, A1b, A2b, GRb)))
        Tel = promote_type(Tel, eltype(GLb), eltype(A1b), eltype(A2b), eltype(GRb))
    end
    for (Ia, A1b) in nonzero_pairs(W1.A), (Ib, B2b) in nonzero_pairs(W2.B)
        Ia[4] == Ib[1] || continue
        a, m = Ia[1], Ia[4]
        GLb = GL2[a]
        nid = Int(_isidentity_A(W1, Ia)) + Int(_isidentity_B(W2, Ib))
        push!(ab, (a, m, A1b, B2b, _task_cost(2, nid, GLb, A1b, B2b)))
        Tel = promote_type(Tel, eltype(GLb), eltype(A1b), eltype(B2b))
    end
    for (Ib, B1b) in nonzero_pairs(W1.B)
        a = Ib[1]
        GLb = GL2[a]
        push!(be, (a, B1b, _task_cost(1, Int(_isidentity_B(W1, Ib)), GLb, B1b)))
        Tel = promote_type(Tel, eltype(GLb), eltype(B1b))
    end
    nleft = length(aa) + length(ab) + length(be)
    rows = sort!(unique!(vcat(Int[t[1] for t in aa], Int[t[1] for t in ab], Int[t[1] for t in be])))
    pairs = sort!(unique!(vcat(
        Tuple{Int, Int}[(t[1], t[2]) for t in aa],
        Tuple{Int, Int}[(t[1], t[2]) for t in ab]
    )))
    (nleft == 0 || nleft == length(rows)) && return _flat_graph_AC2(GL, W1, W2, GR)

    builder = _GraphBuilder(:tree)
    Tslots = Dict(a => _add_slot!(builder) for a in rows)
    Sslots = Dict(p => _add_slot!(builder) for p in pairs)
    Tnodes = Dict{Int, Int}()
    Snodes = Dict{Tuple{Int, Int}, Int}()
    A1_of = Dict{Tuple{Int, Int}, Any}()
    for t in aa
        A1_of[(t[1], t[2])] = t[3]
    end
    for t in ab
        A1_of[(t[1], t[2])] = t[3]
    end
    for a in rows
        Tnodes[a] = _add_node!(builder, :T, numind(GL2[a]), _t_task(GL2[a], Tslots[a]);
            writes_y = false, produces = [Tslots[a]], meta = (; gl = a, gc = 0, mid = 0))
    end

    for (Ic, C1b) in nonzero_pairs(W1.C), (Ia, A2b) in nonzero_pairs(W2.A)
        Ic[3] == Ia[1] || continue
        GRb = GR2[Ia[4]]
        nid = Int(_isidentity_C(W1, Ic)) + Int(_isidentity_A(W2, Ia))
        cost = _task_cost(2, nid, C1b, A2b, GRb)
        slot = _add_slot!(builder)
        _add_node!(builder, :CA, cost, _ca_task(C1b, A2b, GRb, slot);
            produces = [slot], meta = (; gl = 0, gc = Ic[3], mid = Ia[1]))
        Tel = promote_type(Tel, eltype(C1b), eltype(A2b), eltype(GRb))
    end
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ib, B2b) in nonzero_pairs(W2.B)
        Ic[3] == Ib[1] || continue
        nid = Int(_isidentity_C(W1, Ic)) + Int(_isidentity_B(W2, Ib))
        cost = _task_cost(2, nid, C1b, B2b)
        slot = _add_slot!(builder)
        _add_node!(builder, :CB, cost, _cb_task(C1b, B2b, slot);
            produces = [slot], meta = (; gl = 0, gc = Ic[3], mid = Ib[1]))
        Tel = promote_type(Tel, eltype(C1b), eltype(B2b))
    end
    for (Ic, C2b) in nonzero_pairs(W2.C)
        GRb = GR2[Ic[3]]
        cost = _task_cost(1, Int(_isidentity_C(W2, Ic)), C2b, GRb)
        slot = _add_slot!(builder)
        _add_node!(builder, :IC, cost, _ic_task(C2b, GRb, slot);
            produces = [slot], meta = (; gl = 0, gc = Ic[3], mid = 0))
        Tel = promote_type(Tel, eltype(C2b), eltype(GRb))
    end
    Tel = _add_boundary!(builder, Tel, :II, _apply_II!,
        size(W2, 4) == 1 ? missing : transpose(removeunit(GR[1], 2)))
    Tel = _add_boundary!(builder, Tel, :ID, _apply_ID!,
        nonzero_length(W2.D) > 0 ? only(W2.D) : missing)
    Tel = _add_boundary!(builder, Tel, :DE, _apply_DE!,
        nonzero_length(W1.D) > 0 ? only(W1.D) : missing)
    Tel = _add_boundary!(builder, Tel, :EE, _apply_EE!,
        size(W1, 1) == 1 ? missing : removeunit(GL[end], 2))

    for p in pairs
        a, m = p
        Snodes[p] = _add_node!(builder, :S, numind(A1_of[p]),
            _s_task(A1_of[p], Tslots[a], Sslots[p]);
            writes_y = false, deps = [Tnodes[a]], reads = [Tslots[a]],
            produces = [Sslots[p]], meta = (; gl = a, gc = 0, mid = m))
    end
    for (a, B1b, cost) in be
        _add_node!(builder, :BE, cost, _be_tail_task(Tslots[a], B1b);
            deps = [Tnodes[a]], reads = [Tslots[a]], meta = (; gl = a, gc = 0, mid = 0))
    end
    for (a, m, A1b, A2b, GRb, cost) in aa
        p = (a, m)
        slot = _add_slot!(builder)
        _add_node!(builder, :AA, cost, _aa_tail_task(Sslots[p], slot, A2b, GRb);
            deps = [Snodes[p]], reads = [Sslots[p]], produces = [slot],
            meta = (; gl = a, gc = 0, mid = m))
    end
    for (a, m, A1b, B2b, cost) in ab
        p = (a, m)
        _add_node!(builder, :AB, cost, _ab_tail_task(Sslots[p], B2b);
            deps = [Snodes[p]], reads = [Sslots[p]], meta = (; gl = a, gc = 0, mid = m))
    end
    return _freeze_graph(builder;
        rows_T = rows, pairs_S = pairs, nleft = nleft, nmid = length(aa) + length(ab)), Tel
end

struct ThreadedJordanMPO_AC_Hamiltonian <: DerivativeOperator
    plan::_GraphPlan
    promoted_eltype::Type
    state::_MatvecState
    workers::_WorkerResources
    policy::_ExecutionPolicy
end

struct ThreadedJordanMPO_AC2_Hamiltonian <: DerivativeOperator
    plan::_GraphPlan
    promoted_eltype::Type
    state::_MatvecState
    workers::_WorkerResources
    policy::_ExecutionPolicy
end

const _ThreadedHamiltonian =
    Union{ThreadedJordanMPO_AC_Hamiltonian, ThreadedJordanMPO_AC2_Hamiltonian}

function _threaded_AC_hamiltonian(GL, W, GR; policy = _ExecutionPolicy())
    plan, Tel = policy.tree ? _tree_graph_AC(GL, W, GR) : _flat_graph_AC(GL, W, GR)
    return ThreadedJordanMPO_AC_Hamiltonian(
        plan, Tel, _MatvecState(plan), _WorkerResources(), policy)
end

function _threaded_AC2_hamiltonian(GL, W1, W2, GR; policy = _ExecutionPolicy())
    plan, Tel = policy.tree ? _tree_graph_AC2(GL, W1, W2, GR) : _flat_graph_AC2(GL, W1, W2, GR)
    return ThreadedJordanMPO_AC2_Hamiltonian(
        plan, Tel, _MatvecState(plan), _WorkerResources(), policy)
end

function (H::ThreadedJordanMPO_AC_Hamiltonian)(x::MPSTensor)
    T = promote_type(eltype(x), H.promoted_eltype)
    y = vi_zerovector(x, T)
    return _apply_plan!(y, H.plan, H.state, H.workers, H.policy, x)
end
function (H::ThreadedJordanMPO_AC2_Hamiltonian)(x::MPOTensor)
    T = promote_type(eltype(x), H.promoted_eltype)
    y = vi_zerovector(x, T)
    return _apply_plan!(y, H.plan, H.state, H.workers, H.policy, x)
end

"""One-site effective Hamiltonian; falls back to MPSKit when disabled."""
function AC_hamiltonian(
        site::Int, below::FiniteMPS{<:MPSTensor}, operator::MPOHamiltonian{<:JordanMPOTensor},
        above::FiniteMPS{<:MPSTensor}, envs;
        prepare::Bool = true, backend::AbstractBackend = DefaultBackend(),
        allocator = DefaultAllocator(), threaded_policy = _ExecutionPolicy()
    )
    @assert below === above "JordanMPO assumptions break"
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site, below)
    W = operator[site]
    if backend isa DefaultBackend && _policy_cap(threaded_policy) > 0 && nonzero_length(W.A) > 1
        return _threaded_AC_hamiltonian(GL, W, GR; policy = threaded_policy)
    end
    H0 = JordanMPO_AC_Hamiltonian(GL, W, GR; backend, allocator)
    return prepare ? prepare_operator!!(H0) : H0
end

"""Two-site effective Hamiltonian; falls back to MPSKit when disabled."""
function AC2_hamiltonian(
        site::Int, below::FiniteMPS{<:MPSTensor}, operator::MPOHamiltonian{<:JordanMPOTensor},
        above::FiniteMPS{<:MPSTensor}, envs;
        prepare::Bool = true, backend::AbstractBackend = DefaultBackend(),
        allocator = DefaultAllocator(), threaded_policy = _ExecutionPolicy()
    )
    @assert below === above "JordanMPO assumptions break"
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site + 1, below)
    W1, W2 = operator[site], operator[site + 1]
    nA1, nA2 = nonzero_length(W1.A), nonzero_length(W2.A)
    if backend isa DefaultBackend && _policy_cap(threaded_policy) > 0 &&
       nA1 > 0 && nA2 > 0 && nA1 * nA2 > 1
        return _threaded_AC2_hamiltonian(GL, W1, W2, GR; policy = threaded_policy)
    end
    H0 = JordanMPO_AC2_Hamiltonian(GL, W1, W2, GR; backend, allocator)
    return prepare ? prepare_operator!!(H0) : H0
end

_release_hamiltonian!(H) = nothing
function _release_hamiltonian!(H::_ThreadedHamiltonian)
    lock(H.workers.matvec_lock) do
        _release_state!(H.state)
        _release_workers!(H.workers)
    end
    return H
end
Base.empty!(H::_ThreadedHamiltonian) = _release_hamiltonian!(H)

function _set_trace!(H::_ThreadedHamiltonian, enabled::Bool = true)
    lock(H.workers.matvec_lock) do
        H.state.trace_enabled = enabled
    end
    return H
end
function _node_traces(H::_ThreadedHamiltonian)
    lock(H.workers.matvec_lock) do
        return (;
            worker = copy(H.state.trace_worker),
            start_ns = copy(H.state.trace_start),
            stop_ns = copy(H.state.trace_stop),
        )
    end
end
_nodes(plan::_GraphPlan) = plan.nodes
_nodes(H::_ThreadedHamiltonian) = H.plan.nodes

function _cache_stats(state::_MatvecState)
    warm = 0
    bytes = 0
    for value in state.slots
        value === nothing && continue
        warm += 1
        bytes += value isa _StageCache ?
            sum(Base.summarysize, value.tensors) : Base.summarysize(value)
    end
    return warm, bytes
end

function task_report(H::_ThreadedHamiltonian)
    lock(H.workers.matvec_lock) do
        plan = H.plan
        counts = Dict{Symbol, Int}()
        for node in plan.nodes
            counts[node.kind] = get(counts, node.kind, 0) + 1
        end
        gls = [node.meta.gl for node in plan.nodes if node.meta.gl > 0]
        gcs = [node.meta.gc for node in plan.nodes if node.meta.gc > 0]
        warm, cache_bytes = _cache_stats(H.state)
        ws = H.workers
        active = ws.active_workers
        gl_paths = plan.mode === :tree ? plan.nleft : length(gls)
        gl_rows = plan.mode === :tree ? length(plan.rows_T) : length(unique(gls))
        return (;
            ntasks = length(plan.nodes),
            kind_counts = counts,
            mode = plan.mode,
            nroots = length(plan.roots),
            nedges = length(plan.dependents),
            gl_paths,
            gl_rows,
            gc_paths = length(gcs),
            gc_rows = length(unique(gcs)),
            pairs_S = length(plan.pairs_S),
            nmid_paths = plan.nmid,
            gl_redundancy = gl_paths / max(gl_rows, 1),
            s_redundancy = plan.nmid / max(length(plan.pairs_S), 1),
            cache_warm = warm,
            cache_bytes,
            accumulator_bytes = sum(Base.summarysize, ws.buffers; init = 0),
            scratch_bytes = sum(s -> length(s.buffer), ws.scratch; init = 0),
            scratch_highwater = sum(s -> Int(s.max_offset), ws.scratch; init = 0),
            worker_scratch_bytes = [length(s.buffer) for s in ws.scratch[1:active]],
            worker_scratch_highwater = [Int(s.max_offset) for s in ws.scratch[1:active]],
            active_workers = active,
            worker_nrun = copy(ws.nrun[1:active]),
            worker_merge_s = copy(ws.tmerge[1:active]),
            worker_wait_s = Float64.(ws.wait_ns[1:active]) ./ 1e9,
            worker_compute_s = Float64.(ws.compute_ns[1:active]) ./ 1e9,
        )
    end
end

function Base.show(io::IO, H::_ThreadedHamiltonian)
    plan = H.plan
    counts = Dict{Symbol, Int}()
    gls = Int[]
    gcs = Int[]
    for node in plan.nodes
        counts[node.kind] = get(counts, node.kind, 0) + 1
        node.meta.gl > 0 && push!(gls, node.meta.gl)
        node.meta.gc > 0 && push!(gcs, node.meta.gc)
    end
    kinds = sort!(collect(counts); by = last, rev = true)
    kind_str = join(("$k=$n" for (k, n) in kinds), " ")
    gl_paths = plan.mode === :tree ? plan.nleft : length(gls)
    gl_rows = plan.mode === :tree ? length(plan.rows_T) : length(unique(gls))
    return print(io, typeof(H).name.name,
        "($(length(plan.nodes)) $(plan.mode) nodes: $kind_str; " *
        "gl rows $gl_rows/$gl_paths, gc rows $(length(unique(gcs)))/$(length(gcs)))")
end
