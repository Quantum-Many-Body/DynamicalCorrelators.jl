# Threaded path-level evaluation of Jordan-MPO effective Hamiltonians.
#
# MPSKit's default path prepares `GL ⊗ W.A` into a single fused (dense) tensor so
# that each matvec is one big contraction. For Hamiltonians built from sparse
# Jordan MPOs with symmetry sectors, that fusion multiplies a large number of
# exactly-zero blocks. Here every nonzero MPO path is one task, evaluated by
# staged contractions (no dense boundary precontraction); all tasks are sorted
# heavy-first like FiniteMPS.jl.
# This keeps the intermediates environment-sized instead of materializing a
# 6-leg `CA`/`AB` tensor, and avoids concentrating the work in a few
# multi-second tasks.
#
# Execution mirrors FiniteMPS.jl's `action!`:
#   * the serial branch writes every task directly into `y`;
#   * the parallel branch runs `Threads.@threads :greedy` over the flat task
#     list, allocates one fresh output per task, and adds it into `y` under a
#     lock as soon as the task finishes;
#   * all contractions pass `allocator = ManualAllocator()`, so intermediate
#     (and permutation) temporaries are backed by `Libc.malloc`/`free` and
#     bypass the GC entirely. The manual allocator only backs `istemp`
#     intermediates (freed by the @plansor-generated code itself), while task
#     outputs stay GC-managed.
#     MPSKit's scratch BufferAllocator remains unusable here — it is not
#     thread-safe.
#
# Contraction order: left to the TensorOperations optree — at YC-Hubbard sizes
# it matches the hand-picked left-leaning (env·x first) order, while the
# right-leaning tree is substantially slower in local microbenchmarks.
#
# Numerics: identical up to floating-point reduction order (the accumulation
# order over tasks differs from the serial fused path), i.e. ~ulp-level
# differences, as with any parallel reduction.
#
# The `AC_hamiltonian`/`AC2_hamiltonian` methods overloaded here accept MPSKit's
# `prepare`/`backend`/`allocator` keywords so internal MPSKit calls (which always
# pass them) keep working unchanged. The threaded operator is used whenever
# more than one path exists and the setting from `set_threaded_hamiltonian!`
# is nonzero; `1` means serial task evaluation and `0` falls back to MPSKit's
# default fused operator.
#
# BEFORE writing any new contraction here, read
# docs/tensorkit-plansor-conventions.md (the @plansor planar leg-order rules)
# and confirm the index scheme with the user first.

"""
    set_threaded_hamiltonian!(n::Integer) -> Int

Set the maximum number of worker tasks for the threaded effective
Hamiltonians. `n = 1` evaluates the flat task list serially in the calling
task; `n > 1` is capped by `Threads.nthreads()` and by the number of tasks,
and uses the per-worker accumulator pool. `n = 0` disables the threaded
operator entirely, falling back to MPSKit's default fused operator. Returns
the setting.

    set_threaded_hamiltonian!(enable::Bool) -> Int

Legacy toggle: `false` is `n = 0`, `true` is `n = Threads.nthreads()`.
"""
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

# -1 = uninitialized → default `Threads.nthreads()`. A lazy sentinel is needed
# because a `const` initialized with `Threads.nthreads()` at module top level
# would bake in the thread count of the *precompilation* process (usually 1)
# instead of the session's
const _THREADED_HAMILTONIAN_THREADS = Ref{Int}(-1)

_threading_enabled() = threaded_hamiltonian_threads() > 0

# ---------------------------------------------------------------------------
# Optional per-step profiling (serial only)
# ---------------------------------------------------------------------------
#
# The retained scratch-cache kernels below evaluate one MPO path in several
# binary contractions.  For performance work we need the wall time of each
# step, not just the whole task.  The switch is off by default and the hot-path
# cost is one boolean check per step.  The accumulator is a plain Dict and is
# therefore only safe when the task pool runs serially (cap = 1); callers that
# enable it must not run several states concurrently against the same process.
const _PROFILE_STEPS = Ref(false)
const _PROFILE_KIND = Ref(:unknown)
const _PROFILE_STEP_TIMES = Dict{Tuple{Symbol,Symbol},Tuple{Int64,Int64}}()

macro profstep(name, expr)
    quote
        if _PROFILE_STEPS[]
            local t0 = time_ns()
            local val = $(esc(expr))
            local dt = time_ns() - t0
            local key = (_PROFILE_KIND[], $(esc(name)))
            local old = get(_PROFILE_STEP_TIMES, key, (0, 0))
            _PROFILE_STEP_TIMES[key] = (old[1] + dt, old[2] + 1)
            val
        else
            $(esc(expr))
        end
    end
end

function _profile_reset!()
    empty!(_PROFILE_STEP_TIMES)
    return nothing
end

function _profile_enable!(on::Bool)
    _PROFILE_STEPS[] = on
    return on
end

# ---------------------------------------------------------------------------
# boundary blocks as pool tasks
# ---------------------------------------------------------------------------

# In-place two-tensor boundary maps `y += op·x` (the permutation temporaries
# some of them need go through the manual allocator as well).
_apply_onsite!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3] += x[-1 1; -3] * op[-2; 1]; y)
_apply_finished!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3] += op[-1; 1] * x[1 -2; -3]; y)
_apply_notstarted!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3] += x[-1 -2; 1] * op[1; -3]; y)

_apply_II!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] += x[-1 -2; 1 -4] * op[-3; 1]; y)
_apply_ID!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] += x[-1 -2; -3 1] * op[-4; 1]; y)
_apply_DE!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] += x[-1 1; -3 -4] * op[-2; 1]; y)
_apply_EE!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] += x[1 -2; -3 -4] * op[-1; 1]; y)

# ---------------------------------------------------------------------------
# staged boundary kernels (no boundary precontraction)
# ---------------------------------------------------------------------------
#
# Each kernel contracts the environment and the local MPO blocks with `x` in
# several binary steps, mirroring FiniteMPS.jl's `_action!`: the factors are
# never fused into one dense boundary tensor.  The leg orders below were
# verified numerically against the precontracted apply kernels (12-site
# Hubbard, relative errors <= 3.3e-16).

struct _StageCache{T<:Tuple}
    tensors::T
end

function _aa_cache(xproto, GL, A1, A2, GR)
    t1 = @plansor t1[-1 -2; -3 -4 -5] := GL[-1 -2; 1] * xproto[1 -5; -3 -4]
    t2 = @plansor t2[-1 -2; -3 -4 -5] := t1[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
    t3 = @plansor t3[-1 -2; -3 -4 -5] := t2[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
    return _StageCache((t1, t2, t3))
end

function _apply_AA_cached!(y, GL, x, A1, A2, GR, cache::_StageCache{T}) where {T<:Tuple}
    t1, t2, t3 = cache.tensors
    @profstep :GLx begin
        @plansor allocator = ManualAllocator() t1[-1 -2; -3 -4 -5] =
            GL[-1 -2; 1] * x[1 -5; -3 -4]
    end
    @profstep :A1 begin
        @plansor allocator = ManualAllocator() t2[-1 -2; -3 -4 -5] =
            t1[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
    end
    @profstep :A2 begin
        @plansor allocator = ManualAllocator() t3[-1 -2; -3 -4 -5] =
            t2[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
    end
    @profstep :GR begin
        @plansor allocator = ManualAllocator() y[-1 -2; -3 -4] +=
            t3[-1 -2; 1 2 -4] * GR[1 2; -3]
    end
    return y
end

function _ca_cache(xproto, C1, A2, GR)
    t1 = @plansor t1[-1 -2; -3 -4 -5] := xproto[-1 1; -3 -4] * C1[-2; 1 -5]
    t2 = @plansor t2[-1 -2; -3 -4 -5] := t1[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
    return _StageCache((t1, t2))
end

function _apply_CA_cached!(y, x, C1, A2, GR, cache::_StageCache{T}) where {T<:Tuple}
    t1, t2 = cache.tensors
    @profstep :xC1 begin
        @plansor allocator = ManualAllocator() t1[-1 -2; -3 -4 -5] =
            x[-1 1; -3 -4] * C1[-2; 1 -5]
    end
    @profstep :A2 begin
        @plansor allocator = ManualAllocator() t2[-1 -2; -3 -4 -5] =
            t1[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
    end
    @profstep :GR begin
        @plansor allocator = ManualAllocator() y[-1 -2; -3 -4] +=
            t2[-1 -2; 1 2 -4] * GR[1 2; -3]
    end
    return y
end

function _ab_cache(xproto, GL, A1, B2)
    s1 = @plansor s1[-1 -2; -3 -4 -5] := GL[-1 -2; 1] * xproto[1 -5; -3 -4]
    s2 = @plansor s2[-1 -2; -3 -4 -5] := s1[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
    return _StageCache((s1, s2))
end

function _apply_AB_cached!(y, GL, x, A1, B2, cache::_StageCache{T}) where {T<:Tuple}
    s1, s2 = cache.tensors
    @profstep :GLx begin
        @plansor allocator = ManualAllocator() s1[-1 -2; -3 -4 -5] =
            GL[-1 -2; 1] * x[1 -5; -3 -4]
    end
    @profstep :A1 begin
        @plansor allocator = ManualAllocator() s2[-1 -2; -3 -4 -5] =
            s1[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
    end
    @profstep :B2 begin
        @plansor allocator = ManualAllocator() y[-1 -2; -3 -4] +=
            s2[-1 -2; -3 2 1] * B2[1 -4; 2]
    end
    return y
end

function _cb_cache(xproto, C1, B2)
    t = @plansor t[-1 -2; -3 -4 -5] := xproto[-1 1; -3 -4] * C1[-2; 1 -5]
    return _StageCache((t,))
end

function _apply_CB_cached!(y, x, C1, B2, cache::_StageCache{T}) where {T<:Tuple}
    t = cache.tensors[1]
    @profstep :xC1 begin
        @plansor allocator = ManualAllocator() t[-1 -2; -3 -4 -5] =
            x[-1 1; -3 -4] * C1[-2; 1 -5]
    end
    @profstep :B2 begin
        @plansor allocator = ManualAllocator() y[-1 -2; -3 -4] +=
            t[-1 -2; -3 1 2] * B2[2 -4; 1]
    end
    return y
end

function _ic_cache(xproto, C2, GR)
    t = @plansor t[-1 -2; -3 -4 -5] := xproto[-1 -2; -3 1] * C2[-5; 1 -4]
    return _StageCache((t,))
end

function _apply_IC_cached!(y, x, C2, GR, cache::_StageCache{T}) where {T<:Tuple}
    t = cache.tensors[1]
    @profstep :xC2 begin
        @plansor allocator = ManualAllocator() t[-1 -2; -3 -4 -5] =
            x[-1 -2; -3 1] * C2[-5; 1 -4]
    end
    @profstep :GR begin
        @plansor allocator = ManualAllocator() y[-1 -2; -3 -4] +=
            t[-1 -2; 1 2 -4] * GR[1 2; -3]
    end
    return y
end

function _be_cache(xproto, GL, B1)
    t = @plansor t[-1 -2; -3 -4 -5] := GL[-1 -2; 1] * xproto[1 -5; -3 -4]
    return _StageCache((t,))
end

function _apply_BE_cached!(y, GL, x, B1, cache::_StageCache{T}) where {T<:Tuple}
    t = cache.tensors[1]
    @profstep :GLx begin
        @plansor allocator = ManualAllocator() t[-1 -2; -3 -4 -5] =
            GL[-1 -2; 1] * x[1 -5; -3 -4]
    end
    @profstep :B1 begin
        @plansor allocator = ManualAllocator() y[-1 -2; -3 -4] +=
            t[-1 2; -3 -4 1] * B1[2 -2; 1]
    end
    return y
end

function _a_cache(xproto, GL, A, GR)
    t1 = @plansor t1[-1 -2; -3 -4] := GL[-1 -2; 1] * xproto[1 -4; -3]
    t2 = @plansor t2[-1 -2; -3 -4] := t1[-1 2; -3 1] * A[2 -2; 1 -4]
    return _StageCache((t1, t2))
end

function _apply_A_cached!(y, GL, x, A, GR, cache::_StageCache{T}) where {T<:Tuple}
    t1, t2 = cache.tensors
    @profstep :GLx begin
        @plansor allocator = ManualAllocator() t1[-1 -2; -3 -4] =
            GL[-1 -2; 1] * x[1 -4; -3]
    end
    @profstep :A begin
        @plansor allocator = ManualAllocator() t2[-1 -2; -3 -4] =
            t1[-1 2; -3 1] * A[2 -2; 1 -4]
    end
    @profstep :GR begin
        @plansor allocator = ManualAllocator() y[-1 -2; -3] +=
            t2[-1 -2; 1 2] * GR[1 2; -3]
    end
    return y
end

function _b_cache(xproto, GL, B)
    t = @plansor t[-1 -2; -3 -4] := GL[-1 -2; 1] * xproto[1 -4; -3]
    return _StageCache((t,))
end

function _apply_B_cached!(y, GL, x, B, cache::_StageCache{T}) where {T<:Tuple}
    t = cache.tensors[1]
    @profstep :GLx begin
        @plansor allocator = ManualAllocator() t[-1 -2; -3 -4] =
            GL[-1 -2; 1] * x[1 -4; -3]
    end
    @profstep :B begin
        @plansor allocator = ManualAllocator() y[-1 -2; -3] +=
            t[-1 2; -3 1] * B[2 -2; 1]
    end
    return y
end

function _c_cache(xproto, C, GR)
    t = @plansor t[-1 -2; -3 -4] := xproto[-1 1; -3] * C[-2; 1 -4]
    return _StageCache((t,))
end

function _apply_C_cached!(y, x, C, GR, cache::_StageCache{T}) where {T<:Tuple}
    t = cache.tensors[1]
    @profstep :xC begin
        @plansor allocator = ManualAllocator() t[-1 -2; -3 -4] =
            x[-1 1; -3] * C[-2; 1 -4]
    end
    @profstep :GR begin
        @plansor allocator = ManualAllocator() y[-1 -2; -3] +=
            t[-1 -2; 1 2] * GR[1 2; -3]
    end
    return y
end

# Scratch caches are allocated lazily on the first matvec, then reused for the
# remaining Krylov applications. This keeps AC(2)_hamiltonian construction
# cheap (`ham`) without giving up the cached-kernel action time.
@inline function _getcache!(make, cache::Base.RefValue{Any})
    c = cache[]
    if c === nothing
        c = make()
        cache[] = c
    end
    return c
end

function _apply_AA_lazy!(y, GL, x, A1, A2, GR, cache::Base.RefValue{Any})
    c = _getcache!(cache) do
        _aa_cache(x, GL, A1, A2, GR)
    end
    return _apply_AA_cached!(y, GL, x, A1, A2, GR, c)
end
function _apply_CA_lazy!(y, x, C1, A2, GR, cache::Base.RefValue{Any})
    c = _getcache!(cache) do
        _ca_cache(x, C1, A2, GR)
    end
    return _apply_CA_cached!(y, x, C1, A2, GR, c)
end
function _apply_AB_lazy!(y, GL, x, A1, B2, cache::Base.RefValue{Any})
    c = _getcache!(cache) do
        _ab_cache(x, GL, A1, B2)
    end
    return _apply_AB_cached!(y, GL, x, A1, B2, c)
end
function _apply_CB_lazy!(y, x, C1, B2, cache::Base.RefValue{Any})
    c = _getcache!(cache) do
        _cb_cache(x, C1, B2)
    end
    return _apply_CB_cached!(y, x, C1, B2, c)
end
function _apply_IC_lazy!(y, x, C2, GR, cache::Base.RefValue{Any})
    c = _getcache!(cache) do
        _ic_cache(x, C2, GR)
    end
    return _apply_IC_cached!(y, x, C2, GR, c)
end
function _apply_BE_lazy!(y, GL, x, B1, cache::Base.RefValue{Any})
    c = _getcache!(cache) do
        _be_cache(x, GL, B1)
    end
    return _apply_BE_cached!(y, GL, x, B1, c)
end
function _apply_A_lazy!(y, GL, x, A, GR, cache::Base.RefValue{Any})
    c = _getcache!(cache) do
        _a_cache(x, GL, A, GR)
    end
    return _apply_A_cached!(y, GL, x, A, GR, c)
end
function _apply_B_lazy!(y, GL, x, B, cache::Base.RefValue{Any})
    c = _getcache!(cache) do
        _b_cache(x, GL, B)
    end
    return _apply_B_cached!(y, GL, x, B, c)
end
function _apply_C_lazy!(y, x, C, GR, cache::Base.RefValue{Any})
    c = _getcache!(cache) do
        _c_cache(x, C, GR)
    end
    return _apply_C_cached!(y, x, C, GR, c)
end

# Task factories only create the empty cache holder. Binding the factors in
# the factory call also avoids the loop-closure capture trap documented in the
# handover; `xproto` is kept only for call-site compatibility and is unused.
function _aa_task(_xproto, GL, A1, A2, GR)
    return let cache = Ref{Any}(nothing)
        (y, x) -> _apply_AA_lazy!(y, GL, x, A1, A2, GR, cache)
    end
end
function _ca_task(_xproto, C1, A2, GR)
    return let cache = Ref{Any}(nothing)
        (y, x) -> _apply_CA_lazy!(y, x, C1, A2, GR, cache)
    end
end
function _ab_task(_xproto, GL, A1, B2)
    return let cache = Ref{Any}(nothing)
        (y, x) -> _apply_AB_lazy!(y, GL, x, A1, B2, cache)
    end
end
function _cb_task(_xproto, C1, B2)
    return let cache = Ref{Any}(nothing)
        (y, x) -> _apply_CB_lazy!(y, x, C1, B2, cache)
    end
end
function _ic_task(_xproto, C2, GR)
    return let cache = Ref{Any}(nothing)
        (y, x) -> _apply_IC_lazy!(y, x, C2, GR, cache)
    end
end
function _be_task(_xproto, GL, B1)
    return let cache = Ref{Any}(nothing)
        (y, x) -> _apply_BE_lazy!(y, GL, x, B1, cache)
    end
end
function _a_task(_xproto, GL, A, GR)
    return let cache = Ref{Any}(nothing)
        (y, x) -> _apply_A_lazy!(y, GL, x, A, GR, cache)
    end
end
function _b_task(_xproto, GL, B)
    return let cache = Ref{Any}(nothing)
        (y, x) -> _apply_B_lazy!(y, GL, x, B, cache)
    end
end
function _c_task(_xproto, C, GR)
    return let cache = Ref{Any}(nothing)
        (y, x) -> _apply_C_lazy!(y, x, C, GR, cache)
    end
end

struct _CostedTask
    cost::Int
    kind::Symbol
    apply::Function
end

function _push_costed_boundary!(tasks, T, cost, kind::Symbol, apply!, op)
    ismissing(op) && return T
    push!(tasks, _CostedTask(cost, kind, (y, x) -> apply!(y, op, x)))
    return promote_type(T, eltype(op))
end

function _sorted_tasks(tasks)
    sort!(tasks; lt = (a, b) -> a.cost > b.cost)
    return Function[t.apply for t in tasks], Symbol[t.kind for t in tasks]
end

# Execute the flat path-task list with the per-worker accumulator pool.
# The serial branch writes into `y` directly; the parallel branch gives each
# worker one full-size `yl`, distributes tasks through an atomic counter, and
# merges each worker once when it runs out of work.
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
        push!(ws.buffers, VectorInterface.zerovector(y, eltype(y)))
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
                VectorInterface.zerovector!(yl)
                while true
                    i = Threads.atomic_add!(idx, 1)
                    i > total && break
                    btasks[i](yl, x)
                end
                lock(lk) do
                    VectorInterface.add!(y, yl, VectorInterface.One(), VectorInterface.One())
                end
            end
        end
    end
    return y
end

# ---------------------------------------------------------------------------
# threaded effective Hamiltonians
# ---------------------------------------------------------------------------

"""
One-site effective Hamiltonian of a finite Jordan-MPO operator whose boundary
terms are evaluated by staged contractions, one task per nonzero MPO path, and
sorted heavy-first.
"""
struct ThreadedJordanMPO_AC_Hamiltonian <: DerivativeOperator
    btasks::Vector{Function}  # boundary blocks as `(y, x) -> y += ...` tasks
    btask_kinds::Vector{Symbol}  # diagnostic labels parallel to `btasks`
    promoted_eltype::Type     # promote_type over all operator components
    workspace::_AccumulatorWorkspace
end

# Raw constructor used by `AC_hamiltonian`: build the boundary path tensors
# directly from the environment and MPO blocks, without first materializing
# MPSKit's dense boundary contractions (which would duplicate the work).
function _threaded_AC_hamiltonian(
        GL::MPSTensor, W::JordanMPOTensor, GR::MPSTensor, xproto = nothing
    )
    GL2 = GL[2:(end - 1)]
    GR2 = GR[2:(end - 1)]
    tasks = _CostedTask[]
    T = Union{}

    # continuing A
    for (I, Ab) in nonzero_pairs(W.A)
        GLb, GRb = GL2[I[1]], GR2[I[4]]
        push!(tasks, _CostedTask(9, :A, _a_task(xproto, GLb, Ab, GRb)))
        T = promote_type(T, eltype(GLb), eltype(Ab), eltype(GRb))
    end
    # ending B
    for (I, Bb) in nonzero_pairs(W.B)
        GLb = GL2[I[1]]
        push!(tasks, _CostedTask(7, :B, _b_task(xproto, GLb, Bb)))
        T = promote_type(T, eltype(GLb), eltype(Bb))
    end
    # starting C
    for (I, Cb) in nonzero_pairs(W.C)
        GRb = GR2[I[3]]
        push!(tasks, _CostedTask(7, :C, _c_task(xproto, Cb, GRb)))
        T = promote_type(T, eltype(Cb), eltype(GRb))
    end
    # onsite D, finished E, not-started I
    T = _push_costed_boundary!(
        tasks, T, 5, :onsite, _apply_onsite!,
        nonzero_length(W.D) > 0 ? only(W.D) : missing
    )
    T = _push_costed_boundary!(
        tasks, T, 4, :finished, _apply_finished!,
        size(W, 1) == 1 ? missing : removeunit(GL[end], 2)
    )
    T = _push_costed_boundary!(
        tasks, T, 4, :notstarted, _apply_notstarted!,
        size(W, 4) == 1 ? missing : removeunit(GR[1], 2)
    )
    btasks, kinds = _sorted_tasks(tasks)
    ws = _accumulator_workspace()
    return ThreadedJordanMPO_AC_Hamiltonian(btasks, kinds, T, ws)
end

function (H::ThreadedJordanMPO_AC_Hamiltonian)(x::MPSTensor)
    y = VectorInterface.zerovector(x, promote_type(eltype(x), H.promoted_eltype))
    return _execute_tasks!(y, H.btasks, H.workspace, x)
end

"""
Two-site effective Hamiltonian of a finite Jordan-MPO operator whose terms are
evaluated by staged contractions, one task per nonzero MPO path, and sorted
heavy-first.
"""
struct ThreadedJordanMPO_AC2_Hamiltonian <: DerivativeOperator
    btasks::Vector{Function}  # boundary blocks as `(y, x) -> y += ...` tasks
    btask_kinds::Vector{Symbol}  # diagnostic labels parallel to `btasks`
    promoted_eltype::Type
    workspace::_AccumulatorWorkspace
end

# Raw constructor used by `AC2_hamiltonian`.  Every boundary term of MPSKit's
# `JordanMPO_AC2_Hamiltonian` is reproduced here from the original factors:
# `II/ID/DE/EE` stay single blocks, while `IC/CB/CA/AB/BE` become one task per
# nonzero path.  The precontractions are MPSKit's formulas verbatim, so the
# planarity of the resulting tensors is inherited from upstream.
function _threaded_AC2_hamiltonian(
        GL::MPSTensor, W1::JordanMPOTensor, W2::JordanMPOTensor, GR::MPSTensor,
        xproto = nothing
    )
    GL2 = GL[2:(end - 1)]
    GR2 = GR[2:(end - 1)]
    tasks = _CostedTask[]
    sizehint!(tasks, 48)
    T = Union{}

    # continuing left - continuing right
    for (I1, A1b) in nonzero_pairs(W1.A), (I2, A2b) in nonzero_pairs(W2.A)
        I1[4] == I2[1] || continue
        GLb, GRb = GL2[I1[1]], GR2[I2[4]]
        push!(tasks, _CostedTask(12, :AA, _aa_task(xproto, GLb, A1b, A2b, GRb)))
        T = promote_type(T, eltype(GLb), eltype(A1b), eltype(A2b), eltype(GRb))
    end
    # starting left - continuing right
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ia, A2b) in nonzero_pairs(W2.A)
        Ic[3] == Ia[1] || continue
        GRb = GR2[Ia[4]]
        push!(tasks, _CostedTask(10, :CA, _ca_task(xproto, C1b, A2b, GRb)))
        T = promote_type(T, eltype(C1b), eltype(A2b), eltype(GRb))
    end
    # continuing left - ending right
    for (Ia, A1b) in nonzero_pairs(W1.A), (Ib, B2b) in nonzero_pairs(W2.B)
        Ia[4] == Ib[1] || continue
        GLb = GL2[Ia[1]]
        push!(tasks, _CostedTask(10, :AB, _ab_task(xproto, GLb, A1b, B2b)))
        T = promote_type(T, eltype(GLb), eltype(A1b), eltype(B2b))
    end
    # starting left - ending right
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ib, B2b) in nonzero_pairs(W2.B)
        Ic[3] == Ib[1] || continue
        push!(tasks, _CostedTask(8, :CB, _cb_task(xproto, C1b, B2b)))
        T = promote_type(T, eltype(C1b), eltype(B2b))
    end
    # starting right
    for (Ic, C2b) in nonzero_pairs(W2.C)
        GRb = GR2[Ic[3]]
        push!(tasks, _CostedTask(7, :IC, _ic_task(xproto, C2b, GRb)))
        T = promote_type(T, eltype(C2b), eltype(GRb))
    end
    # ending left
    for (Ib, B1b) in nonzero_pairs(W1.B)
        GLb = GL2[Ib[1]]
        push!(tasks, _CostedTask(7, :BE, _be_task(xproto, GLb, B1b)))
        T = promote_type(T, eltype(GLb), eltype(B1b))
    end
    # not started
    T = _push_costed_boundary!(
        tasks, T, 4, :II, _apply_II!,
        size(W2, 4) == 1 ? missing : transpose(removeunit(GR[1], 2))
    )
    # onsite right
    T = _push_costed_boundary!(
        tasks, T, 5, :ID, _apply_ID!,
        nonzero_length(W2.D) > 0 ? only(W2.D) : missing
    )
    # onsite left
    T = _push_costed_boundary!(
        tasks, T, 5, :DE, _apply_DE!,
        nonzero_length(W1.D) > 0 ? only(W1.D) : missing
    )
    # finished
    T = _push_costed_boundary!(
        tasks, T, 4, :EE, _apply_EE!,
        size(W1, 1) == 1 ? missing : removeunit(GL[end], 2)
    )
    btasks, kinds = _sorted_tasks(tasks)
    ws = _accumulator_workspace()
    return ThreadedJordanMPO_AC2_Hamiltonian(btasks, kinds, T, ws)
end

function (H::ThreadedJordanMPO_AC2_Hamiltonian)(x::MPOTensor)
    y = VectorInterface.zerovector(x, promote_type(eltype(x), H.promoted_eltype))
    return _execute_tasks!(y, H.btasks, H.workspace, x)
end

# ---------------------------------------------------------------------------
# AC_hamiltonian / AC2_hamiltonian overloads
# ---------------------------------------------------------------------------

"""
    AC_hamiltonian(site, below, operator::MPOHamiltonian{<:JordanMPOTensor}, above, envs; ...)

Finite Jordan-MPO specialization that evaluates every nonzero MPO path as one
task. Falls back to MPSKit's default operator when threading is disabled or
pointless (single thread or at most one path).
"""
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

"""
    AC2_hamiltonian(site, below, operator::MPOHamiltonian{<:JordanMPOTensor}, above, envs; ...)

Finite Jordan-MPO specialization that evaluates every nonzero MPO path as one
task, with the same fallback rules as the one-site specialization.
"""
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
