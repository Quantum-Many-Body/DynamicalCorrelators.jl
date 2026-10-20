# Threaded path-level evaluation of Jordan-MPO effective Hamiltonians.
# Each nonzero MPO path is one task; workers accumulate into private buffers
# that are merged into the output after the task loop.

"""Set the threaded-Hamiltonian worker cap (0 = fused fallback, 1 = serial)."""
function set_threaded_hamiltonian!(n::Integer)
    n >= 0 || throw(ArgumentError("threaded-hamiltonian cap must be >= 0, got $n"))
    return _THREADED_HAMILTONIAN_THREADS[] = Int(n)
end
set_threaded_hamiltonian!(enable::Bool) =
    set_threaded_hamiltonian!(enable ? Threads.nthreads() : 0)

"""Current worker cap (0 = disabled, 1 = serial, >1 = maximum workers)."""
threaded_hamiltonian_threads() =
    _THREADED_HAMILTONIAN_THREADS[] < 0 ? Threads.nthreads() :
        _THREADED_HAMILTONIAN_THREADS[]

# Lazy sentinel: avoid baking the precompilation process's thread count.
const _THREADED_HAMILTONIAN_THREADS = Ref{Int}(-1)

_threading_enabled() = threaded_hamiltonian_threads() > 0

_apply_onsite(op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3] := x[-1 1; -3] * op[-2; 1]; y)
_apply_finished(op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3] := op[-1; 1] * x[1 -2; -3]; y)
_apply_notstarted(op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3] := x[-1 -2; 1] * op[1; -3]; y)

_apply_II(op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] := x[-1 -2; 1 -4] * op[-3; 1]; y)
_apply_ID(op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] := x[-1 -2; -3 1] * op[-4; 1]; y)
_apply_DE(op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] := x[-1 1; -3 -4] * op[-2; 1]; y)
_apply_EE(op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] := x[1 -2; -3 -4] * op[-1; 1]; y)

struct _ManualCacheAllocator end

const _MANUAL_CACHE_BYTES = Threads.Atomic{Int}(0)

function _cache_payload_bytes(t)
    if t isa AbstractArray
        return length(t) * sizeof(eltype(t))
    elseif hasproperty(t, :data)
        return _cache_payload_bytes(getproperty(t, :data))
    end
    return 0
end

_cache_bytes_delta!(n::Integer) = Threads.atomic_add!(_MANUAL_CACHE_BYTES, Int(n))
manual_cache_bytes() = Int(_MANUAL_CACHE_BYTES[])

function tensoralloc(
        ::Type{A}, structure, ::Val{istemp}, ::_ManualCacheAllocator
    ) where {A<:AbstractArray, istemp}
    a = tensoralloc(A, structure, Val(true), ManualAllocator())
    _cache_bytes_delta!(length(a) * sizeof(eltype(a)))
    return a
end

function tensorfree!(C::AbstractArray, ::_ManualCacheAllocator)
    _cache_bytes_delta!(-length(C) * sizeof(eltype(C)))
    return tensorfree!(C, ManualAllocator())
end

mutable struct _StageCache{T<:Tuple}
    tensors::T
    function _StageCache{T}(tensors::T) where {T<:Tuple}
        cache = new{T}(tensors)
        finalizer(cache) do c
            for t in c.tensors
                _cache_bytes_delta!(-_cache_payload_bytes(t))
                tensorfree!(t, ManualAllocator())
            end
            return nothing
        end
        return cache
    end
end

_stage_cache(tensors::T) where {T<:Tuple} = _StageCache{T}(tensors)

@inline function _getcache!(make, cache::Base.RefValue{Any})
    c = cache[]
    if c === nothing
        c = make()
        cache[] = c
    end
    return c
end

function _a_cache(x, GL, A)
    t1 = @plansor allocator = _ManualCacheAllocator() t1[-1 -2; -3 -4] :=
        GL[-1 -2; 1] * x[1 -4; -3]
    t2 = @plansor allocator = _ManualCacheAllocator() t2[-1 -2; -3 -4] :=
        t1[-1 2; -3 1] * A[2 -2; 1 -4]
    return _stage_cache((t1, t2))
end

function _apply_A_cached!(y, cache, GL, x, A, GR)
    c = _getcache!(cache) do
        _a_cache(x, GL, A)
    end
    t1, t2 = c.tensors
    @plansor allocator = ManualAllocator() t1[-1 -2; -3 -4] =
        GL[-1 -2; 1] * x[1 -4; -3]
    @plansor allocator = ManualAllocator() t2[-1 -2; -3 -4] =
        t1[-1 2; -3 1] * A[2 -2; 1 -4]
    @plansor allocator = ManualAllocator() y[-1 -2; -3] +=
        t2[-1 -2; 1 2] * GR[1 2; -3]
    return y
end

function _b_cache(x, GL)
    t = @plansor allocator = _ManualCacheAllocator() t[-1 -2; -3 -4] :=
        GL[-1 -2; 1] * x[1 -4; -3]
    return _stage_cache((t,))
end

function _apply_B_cached!(y, cache, GL, x, B)
    c = _getcache!(cache) do
        _b_cache(x, GL)
    end
    t = c.tensors[1]
    @plansor allocator = ManualAllocator() t[-1 -2; -3 -4] =
        GL[-1 -2; 1] * x[1 -4; -3]
    @plansor allocator = ManualAllocator() y[-1 -2; -3] +=
        t[-1 2; -3 1] * B[2 -2; 1]
    return y
end

function _c_cache(x, C)
    t = @plansor allocator = _ManualCacheAllocator() t[-1 -2; -3 -4] :=
        x[-1 1; -3] * C[-2; 1 -4]
    return _stage_cache((t,))
end

function _apply_C_cached!(y, cache, x, C, GR)
    c = _getcache!(cache) do
        _c_cache(x, C)
    end
    t = c.tensors[1]
    @plansor allocator = ManualAllocator() t[-1 -2; -3 -4] =
        x[-1 1; -3] * C[-2; 1 -4]
    @plansor allocator = ManualAllocator() y[-1 -2; -3] +=
        t[-1 -2; 1 2] * GR[1 2; -3]
    return y
end

function _cb_cache(x, C1)
    t = @plansor allocator = _ManualCacheAllocator() t[-1 -2; -3 -4 -5] :=
        x[-1 1; -3 -4] * C1[-2; 1 -5]
    return _stage_cache((t,))
end

function _apply_CB_cached!(y, cache, x, C1, B2)
    c = _getcache!(cache) do
        _cb_cache(x, C1)
    end
    t = c.tensors[1]
    @plansor allocator = ManualAllocator() t[-1 -2; -3 -4 -5] =
        x[-1 1; -3 -4] * C1[-2; 1 -5]
    @plansor allocator = ManualAllocator() y[-1 -2; -3 -4] +=
        t[-1 -2; -3 1 2] * B2[2 -4; 1]
    return y
end

function _ic_cache(x, C2)
    t = @plansor allocator = _ManualCacheAllocator() t[-1 -2; -3 -4 -5] :=
        x[-1 -2; -3 1] * C2[-5; 1 -4]
    return _stage_cache((t,))
end

function _apply_IC_cached!(y, cache, x, C2, GR)
    c = _getcache!(cache) do
        _ic_cache(x, C2)
    end
    t = c.tensors[1]
    @plansor allocator = ManualAllocator() t[-1 -2; -3 -4 -5] =
        x[-1 -2; -3 1] * C2[-5; 1 -4]
    @plansor allocator = ManualAllocator() y[-1 -2; -3 -4] +=
        t[-1 -2; 1 2 -4] * GR[1 2; -3]
    return y
end

function _be_cache(x, GL)
    t = @plansor allocator = _ManualCacheAllocator() t[-1 -2; -3 -4 -5] :=
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    return _stage_cache((t,))
end

function _apply_BE_cached!(y, cache, GL, x, B1)
    c = _getcache!(cache) do
        _be_cache(x, GL)
    end
    t = c.tensors[1]
    @plansor allocator = ManualAllocator() t[-1 -2; -3 -4 -5] =
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    @plansor allocator = ManualAllocator() y[-1 -2; -3 -4] +=
        t[-1 2; -3 -4 1] * B1[2 -2; 1]
    return y
end

function _aa_cache(x, GL, A1, A2)
    t1 = @plansor allocator = _ManualCacheAllocator() t1[-1 -2; -3 -4 -5] :=
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    t2 = @plansor allocator = _ManualCacheAllocator() t2[-1 -2; -3 -4 -5] :=
        t1[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
    t3 = @plansor allocator = _ManualCacheAllocator() t3[-1 -2; -3 -4 -5] :=
        t2[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
    return _stage_cache((t1, t2, t3))
end

function _apply_AA_cached!(y, cache, GL, x, A1, A2, GR)
    c = _getcache!(cache) do
        _aa_cache(x, GL, A1, A2)
    end
    t1, t2, t3 = c.tensors
    @plansor allocator = ManualAllocator() t1[-1 -2; -3 -4 -5] =
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    @plansor allocator = ManualAllocator() t2[-1 -2; -3 -4 -5] =
        t1[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
    @plansor allocator = ManualAllocator() t3[-1 -2; -3 -4 -5] =
        t2[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
    @plansor allocator = ManualAllocator() y[-1 -2; -3 -4] +=
        t3[-1 -2; 1 2 -4] * GR[1 2; -3]
    return y
end

function _ca_cache(x, C1, A2)
    t1 = @plansor allocator = _ManualCacheAllocator() t1[-1 -2; -3 -4 -5] :=
        x[-1 1; -3 -4] * C1[-2; 1 -5]
    t2 = @plansor allocator = _ManualCacheAllocator() t2[-1 -2; -3 -4 -5] :=
        t1[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
    return _stage_cache((t1, t2))
end

function _apply_CA_cached!(y, cache, x, C1, A2, GR)
    c = _getcache!(cache) do
        _ca_cache(x, C1, A2)
    end
    t1, t2 = c.tensors
    @plansor allocator = ManualAllocator() t1[-1 -2; -3 -4 -5] =
        x[-1 1; -3 -4] * C1[-2; 1 -5]
    @plansor allocator = ManualAllocator() t2[-1 -2; -3 -4 -5] =
        t1[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
    @plansor allocator = ManualAllocator() y[-1 -2; -3 -4] +=
        t2[-1 -2; 1 2 -4] * GR[1 2; -3]
    return y
end

function _ab_cache(x, GL, A1)
    s1 = @plansor allocator = _ManualCacheAllocator() s1[-1 -2; -3 -4 -5] :=
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    s2 = @plansor allocator = _ManualCacheAllocator() s2[-1 -2; -3 -4 -5] :=
        s1[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
    return _stage_cache((s1, s2))
end

function _apply_AB_cached!(y, cache, GL, x, A1, B2)
    c = _getcache!(cache) do
        _ab_cache(x, GL, A1)
    end
    s1, s2 = c.tensors
    @plansor allocator = ManualAllocator() s1[-1 -2; -3 -4 -5] =
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    @plansor allocator = ManualAllocator() s2[-1 -2; -3 -4 -5] =
        s1[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
    @plansor allocator = ManualAllocator() y[-1 -2; -3 -4] +=
        s2[-1 -2; -3 2 1] * B2[1 -4; 2]
    return y
end

_a_task(GL, A, GR) =
    let cache = Ref{Any}(nothing)
        (y, x) -> _apply_A_cached!(y, cache, GL, x, A, GR)
    end
_b_task(GL, B) =
    let cache = Ref{Any}(nothing)
        (y, x) -> _apply_B_cached!(y, cache, GL, x, B)
    end
_c_task(C, GR) =
    let cache = Ref{Any}(nothing)
        (y, x) -> _apply_C_cached!(y, cache, x, C, GR)
    end
_cb_task(C1, B2) =
    let cache = Ref{Any}(nothing)
        (y, x) -> _apply_CB_cached!(y, cache, x, C1, B2)
    end
_ic_task(C2, GR) =
    let cache = Ref{Any}(nothing)
        (y, x) -> _apply_IC_cached!(y, cache, x, C2, GR)
    end
_be_task(GL, B1) =
    let cache = Ref{Any}(nothing)
        (y, x) -> _apply_BE_cached!(y, cache, GL, x, B1)
    end
_aa_task(GL, A1, A2, GR) =
    let cache = Ref{Any}(nothing)
        (y, x) -> _apply_AA_cached!(y, cache, GL, x, A1, A2, GR)
    end
_ca_task(C1, A2, GR) =
    let cache = Ref{Any}(nothing)
        (y, x) -> _apply_CA_cached!(y, cache, x, C1, A2, GR)
    end
_ab_task(GL, A1, B2) =
    let cache = Ref{Any}(nothing)
        (y, x) -> _apply_AB_cached!(y, cache, GL, x, A1, B2)
    end

struct _CostedTask
    cost::Int
    kind::Symbol
    apply::Function  # AC/AC2: (y, x) -> y; CBE: _ -> term
end

# Estimate a path from its tensor legs, as FiniteMPS's `_countIntr` does:
# non-identity local operators contribute `numind(H) - 2`, identities zero.
_task_cost(noperators::Int, nidentity::Int, tensors...) =
    sum(numind, tensors; init = 0) - 2 * (noperators - nidentity)

_isidentity_A(W, I) =
    haskey(W.scalars, CartesianIndex(I[1] + 1, 1, 1, I[4] + 1))
_isidentity_B(W, I) =
    haskey(W.scalars, CartesianIndex(I[1] + 1, 1, 1, size(W, 4)))
_isidentity_C(W, I) =
    haskey(W.scalars, CartesianIndex(1, 1, 1, I[3] + 1))

function _push_costed_boundary!(
        tasks, T, kind::Symbol, apply, op,
        noperators::Int = 0, nidentity::Int = 0
    )
    ismissing(op) && return T
    cost = _task_cost(noperators, nidentity, op)
    push!(tasks, _CostedTask(cost, kind,
        (y, x) -> (add!(y, apply(op, x), One(), One()); y)))
    return promote_type(T, eltype(op))
end

function _sorted_tasks(tasks)
    # FiniteMPS orders its valid paths once when the projective Hamiltonian is
    # built; action then consumes that ordered list directly.
    perms = sortperm(Int[task.cost for task in tasks]; rev = true)
    return Function[tasks[i].apply for i in perms],
        Symbol[tasks[i].kind for i in perms]
end

mutable struct _AccumulatorWorkspace
    proto::Any
    eltype::Any
    buffers::Vector{Any}
    lock::ReentrantLock
end

_accumulator_workspace() =
    _AccumulatorWorkspace(nothing, nothing, Any[], ReentrantLock())

function _ensure_buffers!(ws::_AccumulatorWorkspace, y, n::Int)
    if ws.proto === nothing || space(ws.proto) != space(y) ||
       eltype(ws.proto) != eltype(y)
        empty!(ws.buffers)
        ws.proto = y
        ws.eltype = eltype(y)
    end
    while length(ws.buffers) < n
        push!(ws.buffers, vi_zerovector(y, eltype(y)))
    end
    return ws
end

function _execute_tasks!(y, btasks, ws::_AccumulatorWorkspace, x)
    total = length(btasks)
    total == 0 && return y
    nt = clamp(min(threaded_hamiltonian_threads(), Threads.nthreads()), 1, total)
    if nt == 1
        for f in btasks
            f(y, x)
        end
        return y
    end

    _ensure_buffers!(ws, y, nt)
    lock(ws.lock) do
        idx = Threads.Atomic{Int}(1)
        lk = ReentrantLock()
        Threads.@sync for w in 1:nt
            Threads.@spawn begin
                yl = ws.buffers[w]
                vi_zerovector!(yl)
                while true
                    i = Threads.atomic_add!(idx, 1)
                    i > total && break
                    btasks[i](yl, x)
                end
                lock(lk) do
                    add!(y, yl, One(), One())
                end
            end
        end
    end
    return y
end

"""One-site effective Hamiltonian."""
struct ThreadedJordanMPO_AC_Hamiltonian <: DerivativeOperator
    btasks::Vector{Function}  # each task is `(y, x) -> y`
    btask_kinds::Vector{Symbol}  # diagnostic labels parallel to `btasks`
    promoted_eltype::Type     # promote_type over all operator components
    workspace::_AccumulatorWorkspace
end

function _threaded_AC_hamiltonian(GL::MPSTensor, W::JordanMPOTensor, GR::MPSTensor)
    GL2 = GL[2:(end - 1)]
    GR2 = GR[2:(end - 1)]
    tasks = _CostedTask[]
    T = Union{}

    for (I, Ab) in nonzero_pairs(W.A)
        GLb, GRb = GL2[I[1]], GR2[I[4]]
        cost = _task_cost(1, Int(_isidentity_A(W, I)), GLb, Ab, GRb)
        push!(tasks, _CostedTask(cost, :A, _a_task(GLb, Ab, GRb)))
        T = promote_type(T, eltype(GLb), eltype(Ab), eltype(GRb))
    end
    for (I, Bb) in nonzero_pairs(W.B)
        GLb = GL2[I[1]]
        cost = _task_cost(1, Int(_isidentity_B(W, I)), GLb, Bb)
        push!(tasks, _CostedTask(cost, :B, _b_task(GLb, Bb)))
        T = promote_type(T, eltype(GLb), eltype(Bb))
    end
    for (I, Cb) in nonzero_pairs(W.C)
        GRb = GR2[I[3]]
        cost = _task_cost(1, Int(_isidentity_C(W, I)), Cb, GRb)
        push!(tasks, _CostedTask(cost, :C, _c_task(Cb, GRb)))
        T = promote_type(T, eltype(Cb), eltype(GRb))
    end
    T = _push_costed_boundary!(
        tasks, T, :onsite, _apply_onsite,
        nonzero_length(W.D) > 0 ? only(W.D) : missing
    )
    T = _push_costed_boundary!(
        tasks, T, :finished, _apply_finished,
        size(W, 1) == 1 ? missing : removeunit(GL[end], 2)
    )
    T = _push_costed_boundary!(
        tasks, T, :notstarted, _apply_notstarted,
        size(W, 4) == 1 ? missing : removeunit(GR[1], 2)
    )
    btasks, kinds = _sorted_tasks(tasks)
    return ThreadedJordanMPO_AC_Hamiltonian(btasks, kinds, T, _accumulator_workspace())
end

function (H::ThreadedJordanMPO_AC_Hamiltonian)(x::MPSTensor)
    T = promote_type(eltype(x), H.promoted_eltype)
    y = vi_zerovector(x, T)
    return _execute_tasks!(y, H.btasks, H.workspace, x)
end

"""Two-site effective Hamiltonian."""
struct ThreadedJordanMPO_AC2_Hamiltonian <: DerivativeOperator
    btasks::Vector{Function}  # each task is `(y, x) -> y`
    btask_kinds::Vector{Symbol}  # diagnostic labels parallel to `btasks`
    promoted_eltype::Type
    workspace::_AccumulatorWorkspace
end

function _threaded_AC2_hamiltonian(
        GL::MPSTensor, W1::JordanMPOTensor, W2::JordanMPOTensor, GR::MPSTensor
    )
    GL2 = GL[2:(end - 1)]
    GR2 = GR[2:(end - 1)]
    tasks = _CostedTask[]
    sizehint!(tasks, 48)
    T = Union{}

    for (I1, A1b) in nonzero_pairs(W1.A), (I2, A2b) in nonzero_pairs(W2.A)
        I1[4] == I2[1] || continue
        GLb, GRb = GL2[I1[1]], GR2[I2[4]]
        nidentity = Int(_isidentity_A(W1, I1)) + Int(_isidentity_A(W2, I2))
        cost = _task_cost(2, nidentity, GLb, A1b, A2b, GRb)
        push!(tasks, _CostedTask(cost, :AA, _aa_task(GLb, A1b, A2b, GRb)))
        T = promote_type(T, eltype(GLb), eltype(A1b), eltype(A2b), eltype(GRb))
    end
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ia, A2b) in nonzero_pairs(W2.A)
        Ic[3] == Ia[1] || continue
        GRb = GR2[Ia[4]]
        nidentity = Int(_isidentity_C(W1, Ic)) + Int(_isidentity_A(W2, Ia))
        cost = _task_cost(2, nidentity, C1b, A2b, GRb)
        push!(tasks, _CostedTask(cost, :CA, _ca_task(C1b, A2b, GRb)))
        T = promote_type(T, eltype(C1b), eltype(A2b), eltype(GRb))
    end
    for (Ia, A1b) in nonzero_pairs(W1.A), (Ib, B2b) in nonzero_pairs(W2.B)
        Ia[4] == Ib[1] || continue
        GLb = GL2[Ia[1]]
        nidentity = Int(_isidentity_A(W1, Ia)) + Int(_isidentity_B(W2, Ib))
        cost = _task_cost(2, nidentity, GLb, A1b, B2b)
        push!(tasks, _CostedTask(cost, :AB, _ab_task(GLb, A1b, B2b)))
        T = promote_type(T, eltype(GLb), eltype(A1b), eltype(B2b))
    end
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ib, B2b) in nonzero_pairs(W2.B)
        Ic[3] == Ib[1] || continue
        nidentity = Int(_isidentity_C(W1, Ic)) + Int(_isidentity_B(W2, Ib))
        cost = _task_cost(2, nidentity, C1b, B2b)
        push!(tasks, _CostedTask(cost, :CB, _cb_task(C1b, B2b)))
        T = promote_type(T, eltype(C1b), eltype(B2b))
    end
    for (Ic, C2b) in nonzero_pairs(W2.C)
        GRb = GR2[Ic[3]]
        cost = _task_cost(1, Int(_isidentity_C(W2, Ic)), C2b, GRb)
        push!(tasks, _CostedTask(cost, :IC, _ic_task(C2b, GRb)))
        T = promote_type(T, eltype(C2b), eltype(GRb))
    end
    for (Ib, B1b) in nonzero_pairs(W1.B)
        GLb = GL2[Ib[1]]
        cost = _task_cost(1, Int(_isidentity_B(W1, Ib)), GLb, B1b)
        push!(tasks, _CostedTask(cost, :BE, _be_task(GLb, B1b)))
        T = promote_type(T, eltype(GLb), eltype(B1b))
    end
    T = _push_costed_boundary!(
        tasks, T, :II, _apply_II,
        size(W2, 4) == 1 ? missing : transpose(removeunit(GR[1], 2))
    )
    T = _push_costed_boundary!(
        tasks, T, :ID, _apply_ID,
        nonzero_length(W2.D) > 0 ? only(W2.D) : missing
    )
    T = _push_costed_boundary!(
        tasks, T, :DE, _apply_DE,
        nonzero_length(W1.D) > 0 ? only(W1.D) : missing
    )
    T = _push_costed_boundary!(
        tasks, T, :EE, _apply_EE,
        size(W1, 1) == 1 ? missing : removeunit(GL[end], 2)
    )
    btasks, kinds = _sorted_tasks(tasks)
    return ThreadedJordanMPO_AC2_Hamiltonian(btasks, kinds, T, _accumulator_workspace())
end

function (H::ThreadedJordanMPO_AC2_Hamiltonian)(x::MPOTensor)
    T = promote_type(eltype(x), H.promoted_eltype)
    y = vi_zerovector(x, T)
    return _execute_tasks!(y, H.btasks, H.workspace, x)
end

"""One-site threaded effective Hamiltonian; falls back to MPSKit when disabled."""
function AC_hamiltonian(
        site::Int, below::FiniteMPS{<:MPSTensor}, operator::MPOHamiltonian{<:JordanMPOTensor},
        above::FiniteMPS{<:MPSTensor}, envs;
        prepare::Bool = true, backend::AbstractBackend = DefaultBackend(),
        allocator = DefaultAllocator()
    )
    @assert below === above "JordanMPO assumptions break"
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site, below)
    W = operator[site]
    if _threading_enabled() && nonzero_length(W.A) > 1
        return _threaded_AC_hamiltonian(GL, W, GR)
    end
    H0 = JordanMPO_AC_Hamiltonian(GL, W, GR; backend, allocator)
    return prepare ? prepare_operator!!(H0) : H0
end

"""Two-site threaded effective Hamiltonian; falls back to MPSKit when disabled."""
function AC2_hamiltonian(
        site::Int, below::FiniteMPS{<:MPSTensor}, operator::MPOHamiltonian{<:JordanMPOTensor},
        above::FiniteMPS{<:MPSTensor}, envs;
        prepare::Bool = true, backend::AbstractBackend = DefaultBackend(),
        allocator = DefaultAllocator()
    )
    @assert below === above "JordanMPO assumptions break"
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site + 1, below)
    W1, W2 = operator[site], operator[site + 1]
    nA1, nA2 = nonzero_length(W1.A), nonzero_length(W2.A)
    if _threading_enabled() && nA1 > 0 && nA2 > 0 && nA1 * nA2 > 1
        return _threaded_AC2_hamiltonian(GL, W1, W2, GR)
    end
    H0 = JordanMPO_AC2_Hamiltonian(GL, W1, W2, GR; backend, allocator)
    return prepare ? prepare_operator!!(H0) : H0
end
