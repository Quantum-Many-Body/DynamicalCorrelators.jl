# Threaded path-level evaluation of Jordan-MPO effective Hamiltonians.
# Each nonzero MPO path is one task; workers accumulate into private buffers
# that are merged into the output after the task loop. Contraction temporaries
# are allocated from a per-worker BufferAllocator (BufferAllocator is not
# thread-safe, so allocators are never shared across workers). Each worker's
# first task overwrites its accumulator (β = 0) instead of zeroing it first.
#
# With the shared-prefix tree enabled (default), left-started paths that share
# an MPO starting row a compute T(a) = GL(a)·x only once, and AA/AB paths
# sharing (a, m) compute S(a, m) = T(a)·A1(a→m) only once. Execution runs in
# three barrier-separated phases — prefixes T, prefixes S (+ BE tails), then
# AA/AB tails — so shared tensors are read-only after their phase and no
# locking or recomputation is needed.

"""
Set the threaded-Hamiltonian worker cap (0 = fused fallback, 1 = serial).
The keyword `tree` toggles the shared-prefix tree for threaded AC2 (left as
is when `nothing`).
"""
function set_threaded_hamiltonian!(n::Integer; tree::Union{Bool, Nothing} = nothing)
    n >= 0 || throw(ArgumentError("threaded-hamiltonian cap must be >= 0, got $n"))
    tree === nothing || (_TREE_ENABLED[] = tree)
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

# Shared-prefix tree for threaded AC2, on by default.
const _TREE_ENABLED = Ref{Bool}(true)
_tree_enabled() = _TREE_ENABLED[]

function _apply_onsite!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3] = x[-1 1; -3] * op[-2; 1]
    else
        @plansor allocator = buffer y[-1 -2; -3] += x[-1 1; -3] * op[-2; 1]
    end
    return y
end
function _apply_finished!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3] = op[-1; 1] * x[1 -2; -3]
    else
        @plansor allocator = buffer y[-1 -2; -3] += op[-1; 1] * x[1 -2; -3]
    end
    return y
end
function _apply_notstarted!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3] = x[-1 -2; 1] * op[1; -3]
    else
        @plansor allocator = buffer y[-1 -2; -3] += x[-1 -2; 1] * op[1; -3]
    end
    return y
end

function _apply_II!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] = x[-1 -2; 1 -4] * op[-3; 1]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] += x[-1 -2; 1 -4] * op[-3; 1]
    end
    return y
end
function _apply_ID!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] = x[-1 -2; -3 1] * op[-4; 1]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] += x[-1 -2; -3 1] * op[-4; 1]
    end
    return y
end
function _apply_DE!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] = x[-1 1; -3 -4] * op[-2; 1]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] += x[-1 1; -3 -4] * op[-2; 1]
    end
    return y
end
function _apply_EE!(y, op, x, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3 -4] = x[1 -2; -3 -4] * op[-1; 1]
    else
        @plansor allocator = buffer y[-1 -2; -3 -4] += x[1 -2; -3 -4] * op[-1; 1]
    end
    return y
end

_task_buffer() = BufferAllocator()

struct _StageCache{T<:Tuple}
    tensors::T
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
    t1 = @plansor allocator = DefaultAllocator() t1[-1 -2; -3 -4] :=
        GL[-1 -2; 1] * x[1 -4; -3]
    t2 = @plansor allocator = DefaultAllocator() t2[-1 -2; -3 -4] :=
        t1[-1 2; -3 1] * A[2 -2; 1 -4]
    return _stage_cache((t1, t2))
end

function _apply_A_cached!(y, cache, buffer, GL, x, A, GR, first::Bool)
    c = _getcache!(cache) do
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
    return y
end

function _b_cache(x, GL)
    t = @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4] :=
        GL[-1 -2; 1] * x[1 -4; -3]
    return _stage_cache((t,))
end

function _apply_B_cached!(y, cache, buffer, GL, x, B, first::Bool)
    c = _getcache!(cache) do
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
    return y
end

function _c_cache(x, C)
    t = @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4] :=
        x[-1 1; -3] * C[-2; 1 -4]
    return _stage_cache((t,))
end

function _apply_C_cached!(y, cache, buffer, x, C, GR, first::Bool)
    c = _getcache!(cache) do
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
    return y
end

function _cb_cache(x, C1)
    t = @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4 -5] :=
        x[-1 1; -3 -4] * C1[-2; 1 -5]
    return _stage_cache((t,))
end

function _apply_CB_cached!(y, cache, buffer, x, C1, B2, first::Bool)
    c = _getcache!(cache) do
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
    return y
end

function _ic_cache(x, C2)
    t = @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4 -5] :=
        x[-1 -2; -3 1] * C2[-5; 1 -4]
    return _stage_cache((t,))
end

function _apply_IC_cached!(y, cache, buffer, x, C2, GR, first::Bool)
    c = _getcache!(cache) do
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
    return y
end

function _be_cache(x, GL)
    t = @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4 -5] :=
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    return _stage_cache((t,))
end

function _apply_BE_cached!(y, cache, buffer, GL, x, B1, first::Bool)
    c = _getcache!(cache) do
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
    return y
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

function _apply_AA_cached!(y, cache, buffer, GL, x, A1, A2, GR, first::Bool)
    c = _getcache!(cache) do
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
    return y
end

function _ca_cache(x, C1, A2)
    t1 = @plansor allocator = DefaultAllocator() t1[-1 -2; -3 -4 -5] :=
        x[-1 1; -3 -4] * C1[-2; 1 -5]
    t2 = @plansor allocator = DefaultAllocator() t2[-1 -2; -3 -4 -5] :=
        t1[-1 -2; -3 2 1] * A2[1 -5; 2 -4]
    return _stage_cache((t1, t2))
end

function _apply_CA_cached!(y, cache, buffer, x, C1, A2, GR, first::Bool)
    c = _getcache!(cache) do
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
    return y
end

function _ab_cache(x, GL, A1)
    s1 = @plansor allocator = DefaultAllocator() s1[-1 -2; -3 -4 -5] :=
        GL[-1 -2; 1] * x[1 -5; -3 -4]
    s2 = @plansor allocator = DefaultAllocator() s2[-1 -2; -3 -4 -5] :=
        s1[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
    return _stage_cache((s1, s2))
end

function _apply_AB_cached!(y, cache, buffer, GL, x, A1, B2, first::Bool)
    c = _getcache!(cache) do
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
    return y
end

# Task factories: the returned closure takes `(y, x, scratch, first::Bool)`;
# the lazily-built stage cache stays with the task (scratch values are
# recomputed every matvec, only the storage is reused).

function _a_task(GL, A, GR)
    cache = Ref{Any}(nothing)
    return (y, x, buf, first) -> _apply_A_cached!(y, cache, buf, GL, x, A, GR, first), cache
end
function _b_task(GL, B)
    cache = Ref{Any}(nothing)
    return (y, x, buf, first) -> _apply_B_cached!(y, cache, buf, GL, x, B, first), cache
end
function _c_task(C, GR)
    cache = Ref{Any}(nothing)
    return (y, x, buf, first) -> _apply_C_cached!(y, cache, buf, x, C, GR, first), cache
end
function _cb_task(C1, B2)
    cache = Ref{Any}(nothing)
    return (y, x, buf, first) -> _apply_CB_cached!(y, cache, buf, x, C1, B2, first), cache
end
function _ic_task(C2, GR)
    cache = Ref{Any}(nothing)
    return (y, x, buf, first) -> _apply_IC_cached!(y, cache, buf, x, C2, GR, first), cache
end
function _be_task(GL, B1)
    cache = Ref{Any}(nothing)
    return (y, x, buf, first) -> _apply_BE_cached!(y, cache, buf, GL, x, B1, first), cache
end
function _aa_task(GL, A1, A2, GR)
    cache = Ref{Any}(nothing)
    return (y, x, buf, first) -> _apply_AA_cached!(y, cache, buf, GL, x, A1, A2, GR, first), cache
end
function _ca_task(C1, A2, GR)
    cache = Ref{Any}(nothing)
    return (y, x, buf, first) -> _apply_CA_cached!(y, cache, buf, x, C1, A2, GR, first), cache
end
function _ab_task(GL, A1, B2)
    cache = Ref{Any}(nothing)
    return (y, x, buf, first) -> _apply_AB_cached!(y, cache, buf, GL, x, A1, B2, first), cache
end

# ---------------------------------------------------------------------------
# Shared-prefix tree tasks. Unlike the flat tasks above, a tree task returns
# a Bool telling the executor whether it wrote into `y` (prefix tasks only
# fill their shared slot and return false). Prefix tensors live in per-row /
# per-pair slots owned by the plan; their storage is built lazily on the
# first matvec and reused afterwards (contents are recomputed every matvec).
# The @plansor index expressions are identical to the flat kernels — AA's t1/
# t2 are exactly T/S, so factored tails reproduce the flat contraction order.
# ---------------------------------------------------------------------------

# T(a) = GL(a)·x, one task per distinct left MPO row a.
function _t_task(GL, slot::Base.RefValue{Any})
    return (y, x, buf, first) -> begin
        T = _getcache!(slot) do
            @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4 -5] :=
                GL[-1 -2; 1] * x[1 -5; -3 -4]
        end
        @plansor allocator = buf T[-1 -2; -3 -4 -5] =
            GL[-1 -2; 1] * x[1 -5; -3 -4]
        return false
    end
end

# S(a, m) = T(a)·A1(a→m), one task per distinct (a, m) pair.
function _s_task(A1, tslot::Base.RefValue{Any}, sslot::Base.RefValue{Any})
    return (y, x, buf, first) -> begin
        Tm = tslot[]
        S = _getcache!(sslot) do
            @plansor allocator = DefaultAllocator() s[-1 -2; -3 -4 -5] :=
                Tm[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
        end
        @plansor allocator = buf S[-1 -2; -3 -4 -5] =
            Tm[-1 1; -3 -4 2] * A1[1 -2; 2 -5]
        return false
    end
end

# AA tail: t3 = S·A2 (per-path cache), then y (+)= t3·GR.
function _aa_tail_task(sslot::Base.RefValue{Any}, A2, GR)
    c3 = Ref{Any}(nothing)
    t = (y, x, buf, first) -> begin
        S = sslot[]
        t3 = _getcache!(c3) do
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
        return true
    end
    return t, c3
end

# AB tail: y (+)= S·B2, no intermediate.
function _ab_tail_task(sslot::Base.RefValue{Any}, B2)
    return (y, x, buf, first) -> begin
        S = sslot[]
        if first
            @plansor allocator = buf y[-1 -2; -3 -4] =
                S[-1 -2; -3 2 1] * B2[1 -4; 2]
        else
            @plansor allocator = buf y[-1 -2; -3 -4] +=
                S[-1 -2; -3 2 1] * B2[1 -4; 2]
        end
        return true
    end
end

# BE tail: y (+)= T·B1, no intermediate.
function _be_tail_task(tslot::Base.RefValue{Any}, B1)
    return (y, x, buf, first) -> begin
        T = tslot[]
        if first
            @plansor allocator = buf y[-1 -2; -3 -4] =
                T[-1 2; -3 -4 1] * B1[2 -2; 1]
        else
            @plansor allocator = buf y[-1 -2; -3 -4] +=
                T[-1 2; -3 -4 1] * B1[2 -2; 1]
        end
        return true
    end
end

# Wrap a flat `(y, x, buf, first) -> y` task for the tree executor.
_writes_y(f) = (y, x, buf, first) -> (f(y, x, buf, first); true)

# One-site (AC) variants: T(a) = GL(a)·x with one physical leg; A tails run
# t2 = T·A (per-path cache) then y (+)= t2·GR; B tails run y (+)= T·B1.
function _t_task_AC(GL, slot::Base.RefValue{Any})
    return (y, x, buf, first) -> begin
        T = _getcache!(slot) do
            @plansor allocator = DefaultAllocator() t[-1 -2; -3 -4] :=
                GL[-1 -2; 1] * x[1 -4; -3]
        end
        @plansor allocator = buf T[-1 -2; -3 -4] =
            GL[-1 -2; 1] * x[1 -4; -3]
        return false
    end
end

function _a_tail_task_AC(tslot::Base.RefValue{Any}, A, GR)
    c2 = Ref{Any}(nothing)
    t = (y, x, buf, first) -> begin
        T = tslot[]
        t2 = _getcache!(c2) do
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
        return true
    end
    return t, c2
end

function _b_tail_task_AC(tslot::Base.RefValue{Any}, B)
    return (y, x, buf, first) -> begin
        T = tslot[]
        if first
            @plansor allocator = buf y[-1 -2; -3] =
                T[-1 2; -3 1] * B[2 -2; 1]
        else
            @plansor allocator = buf y[-1 -2; -3] +=
                T[-1 2; -3 1] * B[2 -2; 1]
        end
        return true
    end
end

struct _CostedTask
    cost::Int
    kind::Symbol
    apply::Function   # (y, x, scratch, first::Bool) -> y
    meta::NamedTuple  # diagnostics: gl/gc MPO block rows + stage-cache holder
end

# Boundary/cheap paths carry no block rows and no stage cache.
const _EMPTY_META = (; gl = 0, gc = 0, cache = nothing)

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
        (y, x, buf, first) -> apply(y, op, x, buf, first), _EMPTY_META))
    return promote_type(T, eltype(op))
end

function _sorted_tasks(tasks)
    # FiniteMPS orders its valid paths once when the projective Hamiltonian is
    # built; action then consumes that ordered list directly.
    perms = sortperm(Int[task.cost for task in tasks]; rev = true)
    return tasks[perms]
end

mutable struct _AccumulatorWorkspace
    proto::Any
    buffers::Vector{Any}              # per-worker accumulators (parallel only)
    scratch::Vector{BufferAllocator}  # per-worker contraction temporaries
    tmerge::Vector{Float64}           # per-worker merge time, last matvec (s)
    nrun::Vector{Int}                 # per-worker executed task count, last matvec
    wrote::Vector{Bool}               # tree mode: worker already wrote its accumulator
    lock::ReentrantLock
end

_accumulator_workspace() =
    _AccumulatorWorkspace(nothing, Any[], BufferAllocator[],
        Float64[], Int[], Bool[], ReentrantLock())

function _ensure_buffers!(ws::_AccumulatorWorkspace, y, n::Int, accumulators::Bool)
    if ws.proto === nothing || space(ws.proto) != space(y) ||
       eltype(ws.proto) != eltype(y)
        empty!(ws.buffers)
        ws.proto = y
    end
    if accumulators
        while length(ws.buffers) < n
            push!(ws.buffers, vi_zerovector(y, eltype(y)))
        end
    end
    while length(ws.scratch) < n
        push!(ws.scratch, _task_buffer())
    end
    if length(ws.tmerge) < n
        append!(ws.tmerge, zeros(n - length(ws.tmerge)))
        append!(ws.nrun, zeros(Int, n - length(ws.nrun)))
        append!(ws.wrote, falses(n - length(ws.wrote)))
    end
    return ws
end

function _execute_tasks!(y, tasks::Vector{_CostedTask}, ws::_AccumulatorWorkspace, x)
    total = length(tasks)
    total == 0 && return y
    nt = clamp(min(threaded_hamiltonian_threads(), Threads.nthreads()), 1, total)
    if nt == 1
        _ensure_buffers!(ws, y, 1, false)
        # NB: keep this local name distinct from the spawned workers' `buf`
        # below — a shared name would make the @spawn closures capture one
        # boxed function-level variable instead of per-worker locals.
        serial_buf = ws.scratch[1]
        for task in tasks
            task.apply(y, x, serial_buf, false)
        end
        return y
    end

    _ensure_buffers!(ws, y, nt, true)
    fill!(ws.tmerge, 0.0)
    fill!(ws.nrun, 0)
    lock(ws.lock) do
        idx = Threads.Atomic{Int}(1)
        lk = ReentrantLock()
        Threads.@sync for w in 1:nt
            Threads.@spawn begin
                yl = ws.buffers[w]
                buf = ws.scratch[w]
                first = true
                n = 0
                while true
                    i = Threads.atomic_add!(idx, 1)
                    i > total && break
                    tasks[i].apply(yl, x, buf, first)
                    first = false
                    n += 1
                end
                ws.nrun[w] = n
                if n > 0
                    t0 = time_ns()
                    lock(lk) do
                        add!(y, yl, One(), One())
                    end
                    ws.tmerge[w] = (time_ns() - t0) / 1e9
                end
            end
        end
    end
    return y
end

# ---------------------------------------------------------------------------
# Shared-prefix tree for the two-site Hamiltonian. The plan groups the
# left-started paths by their MPO starting row a (prefix T(a) = GL(a)·x) and
# the AA/AB paths by (a, m) (prefix S(a, m) = T(a)·A1(a→m)); execution runs
# phase A (T + right-side + boundary), phase B (S + BE tails), phase C (AA/AB
# tails), so every shared tensor is written before the barrier that publishes
# it and is read-only afterwards.
# ---------------------------------------------------------------------------

struct _TreePlan
    phaseA::Vector{_CostedTask}  # :T row prefixes + right-side + boundary
    phaseB::Vector{_CostedTask}  # :S pair prefixes + BE tails
    phaseC::Vector{_CostedTask}  # AA/AB tails
    caches::Vector{Base.RefValue{Any}}  # every lazily filled slot
    proto::Ref{Any}              # x prototype; space/eltype change resets slots
    rows_T::Vector{Int}          # distinct left MPO rows
    pairs_S::Vector{Tuple{Int, Int}}  # distinct (row, mid) pairs
    nleft::Int                   # left-started paths before sharing
    nmid::Int                    # AA+AB paths before sharing
end

# Gate-first ordering: prefix tasks (:T/:S) feed later phases, so they run
# before anything else; within a group, heavier tasks first (as _sorted_tasks).
function _sorted_tree_tasks(tasks)
    perms = sortperm(tasks;
                     by = t -> (t.kind === :T || t.kind === :S, t.cost), rev = true)
    return tasks[perms]
end

function _push_tree_boundary!(tasks, Tel, kind::Symbol, apply, op,
        noperators::Int = 0, nidentity::Int = 0)
    ismissing(op) && return Tel
    cost = _task_cost(noperators, nidentity, op)
    push!(tasks, _CostedTask(cost, kind,
        (y, x, buf, first) -> (apply(y, op, x, buf, first); true), _EMPTY_META))
    return promote_type(Tel, eltype(op))
end

# Pure-structure grouping; returns (plan, promoted_eltype), or (nothing, Tel)
# when no row is shared by two paths (the caller then builds the flat list).
function _tree_plan_AC2(GL::MPSTensor, W1::JordanMPOTensor, W2::JordanMPOTensor, GR::MPSTensor)
    GL2 = GL[2:(end - 1)]
    GR2 = GR[2:(end - 1)]

    aa = Tuple{Int, Int, Any, Any, Any, Int}[]  # a, m, A1b, A2b, GRb, cost
    ab = Tuple{Int, Int, Any, Any, Int}[]       # a, m, A1b, B2b, cost
    be = Tuple{Int, Any, Int}[]                 # a, B1b, cost
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
    rows = sort!(unique!(vcat(Int[t[1] for t in aa], Int[t[1] for t in ab],
                              Int[t[1] for t in be])))
    pairs = sort!(unique!(vcat(Tuple{Int, Int}[(t[1], t[2]) for t in aa],
                               Tuple{Int, Int}[(t[1], t[2]) for t in ab])))
    (nleft == 0 || nleft == length(rows)) && return nothing, Tel

    Tslots = [Ref{Any}(nothing) for _ in rows]
    Sslots = [Ref{Any}(nothing) for _ in pairs]
    row_slot = Dict(a => i for (i, a) in enumerate(rows))
    pair_slot = Dict(p => i for (i, p) in enumerate(pairs))
    A1_of = Dict{Tuple{Int, Int}, Any}()
    for t in aa
        A1_of[(t[1], t[2])] = t[3]
    end
    for t in ab
        A1_of[(t[1], t[2])] = t[3]
    end

    caches = Base.RefValue{Any}[Tslots..., Sslots...]
    phaseA = _CostedTask[]
    phaseB = _CostedTask[]
    phaseC = _CostedTask[]
    sizehint!(phaseA, length(rows) + 16)
    sizehint!(phaseB, length(pairs) + length(be))
    sizehint!(phaseC, length(aa) + length(ab))

    for (i, a) in enumerate(rows)
        push!(phaseA, _CostedTask(numind(GL2[a]), :T, _t_task(GL2[a], Tslots[i]),
                                  (; gl = a, gc = 0, cache = nothing)))
    end
    for (i, p) in enumerate(pairs)
        push!(phaseB, _CostedTask(numind(A1_of[p]), :S,
                                  _s_task(A1_of[p], Tslots[row_slot[p[1]]], Sslots[i]),
                                  (; gl = p[1], gc = 0, cache = nothing)))
    end
    for (a, B1b, cost) in be
        push!(phaseB, _CostedTask(cost, :BE,
                                  _be_tail_task(Tslots[row_slot[a]], B1b),
                                  (; gl = a, gc = 0, cache = nothing)))
    end
    for (a, m, A1b, A2b, GRb, cost) in aa
        t, c3 = _aa_tail_task(Sslots[pair_slot[(a, m)]], A2b, GRb)
        push!(caches, c3)
        push!(phaseC, _CostedTask(cost, :AA, t, (; gl = a, gc = 0, cache = c3)))
    end
    for (a, m, A1b, B2b, cost) in ab
        push!(phaseC, _CostedTask(cost, :AB,
                                  _ab_tail_task(Sslots[pair_slot[(a, m)]], B2b),
                                  (; gl = a, gc = 0, cache = nothing)))
    end

    # Right-side and boundary tasks have no T/S dependency; they ride along in
    # phase A to fill its workers.
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ia, A2b) in nonzero_pairs(W2.A)
        Ic[3] == Ia[1] || continue
        GRb = GR2[Ia[4]]
        nid = Int(_isidentity_C(W1, Ic)) + Int(_isidentity_A(W2, Ia))
        cost = _task_cost(2, nid, C1b, A2b, GRb)
        f, c = _ca_task(C1b, A2b, GRb)
        push!(caches, c)
        push!(phaseA, _CostedTask(cost, :CA, _writes_y(f),
                                  (; gl = 0, gc = Ic[3], cache = c)))
        Tel = promote_type(Tel, eltype(C1b), eltype(A2b), eltype(GRb))
    end
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ib, B2b) in nonzero_pairs(W2.B)
        Ic[3] == Ib[1] || continue
        nid = Int(_isidentity_C(W1, Ic)) + Int(_isidentity_B(W2, Ib))
        cost = _task_cost(2, nid, C1b, B2b)
        f, c = _cb_task(C1b, B2b)
        push!(caches, c)
        push!(phaseA, _CostedTask(cost, :CB, _writes_y(f),
                                  (; gl = 0, gc = Ic[3], cache = c)))
        Tel = promote_type(Tel, eltype(C1b), eltype(B2b))
    end
    for (Ic, C2b) in nonzero_pairs(W2.C)
        GRb = GR2[Ic[3]]
        cost = _task_cost(1, Int(_isidentity_C(W2, Ic)), C2b, GRb)
        f, c = _ic_task(C2b, GRb)
        push!(caches, c)
        push!(phaseA, _CostedTask(cost, :IC, _writes_y(f),
                                  (; gl = 0, gc = Ic[3], cache = c)))
        Tel = promote_type(Tel, eltype(C2b), eltype(GRb))
    end
    Tel = _push_tree_boundary!(phaseA, Tel, :II, _apply_II!,
        size(W2, 4) == 1 ? missing : transpose(removeunit(GR[1], 2)))
    Tel = _push_tree_boundary!(phaseA, Tel, :ID, _apply_ID!,
        nonzero_length(W2.D) > 0 ? only(W2.D) : missing)
    Tel = _push_tree_boundary!(phaseA, Tel, :DE, _apply_DE!,
        nonzero_length(W1.D) > 0 ? only(W1.D) : missing)
    Tel = _push_tree_boundary!(phaseA, Tel, :EE, _apply_EE!,
        size(W1, 1) == 1 ? missing : removeunit(GL[end], 2))

    plan = _TreePlan(_sorted_tree_tasks(phaseA), _sorted_tree_tasks(phaseB),
                     _sorted_tree_tasks(phaseC), caches, Ref{Any}(nothing),
                     rows, pairs, nleft, length(aa) + length(ab))
    return plan, Tel
end

function _ensure_tree!(plan::_TreePlan, x)
    p = plan.proto[]
    if p === nothing || space(p) != space(x) || eltype(p) != eltype(x)
        for c in plan.caches
            c[] = nothing
        end
        plan.proto[] = x
    end
    return plan
end

# One-site analogue: A/B channels share T(a) = GL(a)·x per starting row;
# two phases (A: T + C + boundary, B: A/B tails), phaseC stays empty.
function _tree_plan_AC(GL::MPSTensor, W::JordanMPOTensor, GR::MPSTensor)
    GL2 = GL[2:(end - 1)]
    GR2 = GR[2:(end - 1)]

    aa = Tuple{Int, Any, Any, Int}[]  # a, Ab, GRb, cost
    be = Tuple{Int, Any, Int}[]       # a, Bb, cost
    Tel = Union{}
    for (I, Ab) in nonzero_pairs(W.A)
        a, b = I[1], I[4]
        GLb, GRb = GL2[a], GR2[b]
        push!(aa, (a, Ab, GRb,
                   _task_cost(1, Int(_isidentity_A(W, I)), GLb, Ab, GRb)))
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
    (nleft == 0 || nleft == length(rows)) && return nothing, Tel

    Tslots = [Ref{Any}(nothing) for _ in rows]
    row_slot = Dict(a => i for (i, a) in enumerate(rows))
    caches = Base.RefValue{Any}[Tslots...]
    phaseA = _CostedTask[]
    phaseB = _CostedTask[]
    sizehint!(phaseA, length(rows) + 8)
    sizehint!(phaseB, nleft)

    for (i, a) in enumerate(rows)
        push!(phaseA, _CostedTask(numind(GL2[a]), :T,
                                  _t_task_AC(GL2[a], Tslots[i]),
                                  (; gl = a, gc = 0, cache = nothing)))
    end
    for (a, Ab, GRb, cost) in aa
        t, c2 = _a_tail_task_AC(Tslots[row_slot[a]], Ab, GRb)
        push!(caches, c2)
        push!(phaseB, _CostedTask(cost, :A, t, (; gl = a, gc = 0, cache = c2)))
    end
    for (a, Bb, cost) in be
        push!(phaseB, _CostedTask(cost, :B,
                                  _b_tail_task_AC(Tslots[row_slot[a]], Bb),
                                  (; gl = a, gc = 0, cache = nothing)))
    end
    for (I, Cb) in nonzero_pairs(W.C)
        GRb = GR2[I[3]]
        cost = _task_cost(1, Int(_isidentity_C(W, I)), Cb, GRb)
        f, c = _c_task(Cb, GRb)
        push!(caches, c)
        push!(phaseA, _CostedTask(cost, :C, _writes_y(f),
                                  (; gl = 0, gc = I[3], cache = c)))
        Tel = promote_type(Tel, eltype(Cb), eltype(GRb))
    end
    Tel = _push_tree_boundary!(phaseA, Tel, :onsite, _apply_onsite!,
        nonzero_length(W.D) > 0 ? only(W.D) : missing)
    Tel = _push_tree_boundary!(phaseA, Tel, :finished, _apply_finished!,
        size(W, 1) == 1 ? missing : removeunit(GL[end], 2))
    Tel = _push_tree_boundary!(phaseA, Tel, :notstarted, _apply_notstarted!,
        size(W, 4) == 1 ? missing : removeunit(GR[1], 2))

    plan = _TreePlan(_sorted_tree_tasks(phaseA), _sorted_tree_tasks(phaseB),
                     _CostedTask[], caches, Ref{Any}(nothing), rows,
                     Tuple{Int, Int}[], nleft, 0)
    return plan, Tel
end


# Three barrier-separated phases; accumulators persist across phases and are
# merged once at the end. NB: variables assigned inside the @spawn closures
# (yl, wbuf, n, i, did) must never share a name with any local of this
# function — a shared name would box one function-level variable into all
# closures (the stage-1 bug).
function _execute_tree!(y, plan::_TreePlan, ws::_AccumulatorWorkspace, x)
    _ensure_tree!(plan, x)
    phases = (plan.phaseA, plan.phaseB, plan.phaseC)
    nmax = max(length(plan.phaseA), length(plan.phaseB), length(plan.phaseC))
    nmax == 0 && return y
    nt = clamp(min(threaded_hamiltonian_threads(), Threads.nthreads()), 1, nmax)
    if nt == 1
        _ensure_buffers!(ws, y, 1, false)
        sbuf = ws.scratch[1]
        for tasks in phases, task in tasks
            task.apply(y, x, sbuf, false)
        end
        return y
    end
    _ensure_buffers!(ws, y, nt, true)
    fill!(ws.tmerge, 0.0)
    fill!(ws.nrun, 0)
    fill!(ws.wrote, false)
    lock(ws.lock) do
        for tasks in phases
            isempty(tasks) && continue
            total = length(tasks)
            idx = Threads.Atomic{Int}(1)
            Threads.@sync for w in 1:nt
                Threads.@spawn begin
                    yl = ws.buffers[w]
                    wbuf = ws.scratch[w]
                    n = 0
                    while true
                        i = Threads.atomic_add!(idx, 1)
                        i > total && break
                        did = tasks[i].apply(yl, x, wbuf, !ws.wrote[w])
                        if did
                            ws.wrote[w] = true
                            n += 1
                        end
                    end
                    ws.nrun[w] += n
                end
            end
        end
    end
    for w in 1:nt
        ws.wrote[w] || continue
        t0 = time_ns()
        add!(y, ws.buffers[w], One(), One())
        ws.tmerge[w] = (time_ns() - t0) / 1e9
    end
    return y
end

"""One-site effective Hamiltonian."""
struct ThreadedJordanMPO_AC_Hamiltonian <: DerivativeOperator
    tasks::Vector{_CostedTask}  # each task is `(y, x, scratch, first) -> y`
    plan::Any                   # _TreePlan when the shared-prefix tree applies
    promoted_eltype::Type       # promote_type over all operator components
    workspace::_AccumulatorWorkspace
end

function _threaded_AC_hamiltonian(GL::MPSTensor, W::JordanMPOTensor, GR::MPSTensor)
    if _tree_enabled()
        plan, Tel = _tree_plan_AC(GL, W, GR)
        plan !== nothing && return ThreadedJordanMPO_AC_Hamiltonian(
            _CostedTask[], plan, Tel, _accumulator_workspace())
    end
    GL2 = GL[2:(end - 1)]
    GR2 = GR[2:(end - 1)]
    tasks = _CostedTask[]
    T = Union{}

    for (I, Ab) in nonzero_pairs(W.A)
        GLb, GRb = GL2[I[1]], GR2[I[4]]
        cost = _task_cost(1, Int(_isidentity_A(W, I)), GLb, Ab, GRb)
        t, cache = _a_task(GLb, Ab, GRb)
        push!(tasks, _CostedTask(cost, :A, t, (; gl = I[1], gc = 0, cache)))
        T = promote_type(T, eltype(GLb), eltype(Ab), eltype(GRb))
    end
    for (I, Bb) in nonzero_pairs(W.B)
        GLb = GL2[I[1]]
        cost = _task_cost(1, Int(_isidentity_B(W, I)), GLb, Bb)
        t, cache = _b_task(GLb, Bb)
        push!(tasks, _CostedTask(cost, :B, t, (; gl = I[1], gc = 0, cache)))
        T = promote_type(T, eltype(GLb), eltype(Bb))
    end
    for (I, Cb) in nonzero_pairs(W.C)
        GRb = GR2[I[3]]
        cost = _task_cost(1, Int(_isidentity_C(W, I)), Cb, GRb)
        t, cache = _c_task(Cb, GRb)
        push!(tasks, _CostedTask(cost, :C, t, (; gl = 0, gc = I[3], cache)))
        T = promote_type(T, eltype(Cb), eltype(GRb))
    end
    T = _push_costed_boundary!(
        tasks, T, :onsite, _apply_onsite!,
        nonzero_length(W.D) > 0 ? only(W.D) : missing
    )
    T = _push_costed_boundary!(
        tasks, T, :finished, _apply_finished!,
        size(W, 1) == 1 ? missing : removeunit(GL[end], 2)
    )
    T = _push_costed_boundary!(
        tasks, T, :notstarted, _apply_notstarted!,
        size(W, 4) == 1 ? missing : removeunit(GR[1], 2)
    )
    tasks = _sorted_tasks(tasks)
    return ThreadedJordanMPO_AC_Hamiltonian(tasks, nothing, T, _accumulator_workspace())
end

function (H::ThreadedJordanMPO_AC_Hamiltonian)(x::MPSTensor)
    T = promote_type(eltype(x), H.promoted_eltype)
    y = vi_zerovector(x, T)
    H.plan !== nothing && return _execute_tree!(y, H.plan, H.workspace, x)
    return _execute_tasks!(y, H.tasks, H.workspace, x)
end

"""Two-site effective Hamiltonian."""
struct ThreadedJordanMPO_AC2_Hamiltonian <: DerivativeOperator
    tasks::Vector{_CostedTask}  # each task is `(y, x, scratch, first) -> y`
    plan::Any                   # _TreePlan when the shared-prefix tree applies
    promoted_eltype::Type
    workspace::_AccumulatorWorkspace
end

function _threaded_AC2_hamiltonian(
        GL::MPSTensor, W1::JordanMPOTensor, W2::JordanMPOTensor, GR::MPSTensor
    )
    if _tree_enabled()
        plan, Tel = _tree_plan_AC2(GL, W1, W2, GR)
        plan !== nothing && return ThreadedJordanMPO_AC2_Hamiltonian(
            _CostedTask[], plan, Tel, _accumulator_workspace())
    end
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
        t, cache = _aa_task(GLb, A1b, A2b, GRb)
        push!(tasks, _CostedTask(cost, :AA, t, (; gl = I1[1], gc = 0, cache)))
        T = promote_type(T, eltype(GLb), eltype(A1b), eltype(A2b), eltype(GRb))
    end
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ia, A2b) in nonzero_pairs(W2.A)
        Ic[3] == Ia[1] || continue
        GRb = GR2[Ia[4]]
        nidentity = Int(_isidentity_C(W1, Ic)) + Int(_isidentity_A(W2, Ia))
        cost = _task_cost(2, nidentity, C1b, A2b, GRb)
        t, cache = _ca_task(C1b, A2b, GRb)
        push!(tasks, _CostedTask(cost, :CA, t, (; gl = 0, gc = Ic[3], cache)))
        T = promote_type(T, eltype(C1b), eltype(A2b), eltype(GRb))
    end
    for (Ia, A1b) in nonzero_pairs(W1.A), (Ib, B2b) in nonzero_pairs(W2.B)
        Ia[4] == Ib[1] || continue
        GLb = GL2[Ia[1]]
        nidentity = Int(_isidentity_A(W1, Ia)) + Int(_isidentity_B(W2, Ib))
        cost = _task_cost(2, nidentity, GLb, A1b, B2b)
        t, cache = _ab_task(GLb, A1b, B2b)
        push!(tasks, _CostedTask(cost, :AB, t, (; gl = Ia[1], gc = 0, cache)))
        T = promote_type(T, eltype(GLb), eltype(A1b), eltype(B2b))
    end
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ib, B2b) in nonzero_pairs(W2.B)
        Ic[3] == Ib[1] || continue
        nidentity = Int(_isidentity_C(W1, Ic)) + Int(_isidentity_B(W2, Ib))
        cost = _task_cost(2, nidentity, C1b, B2b)
        t, cache = _cb_task(C1b, B2b)
        push!(tasks, _CostedTask(cost, :CB, t, (; gl = 0, gc = Ic[3], cache)))
        T = promote_type(T, eltype(C1b), eltype(B2b))
    end
    for (Ic, C2b) in nonzero_pairs(W2.C)
        GRb = GR2[Ic[3]]
        cost = _task_cost(1, Int(_isidentity_C(W2, Ic)), C2b, GRb)
        t, cache = _ic_task(C2b, GRb)
        push!(tasks, _CostedTask(cost, :IC, t, (; gl = 0, gc = Ic[3], cache)))
        T = promote_type(T, eltype(C2b), eltype(GRb))
    end
    for (Ib, B1b) in nonzero_pairs(W1.B)
        GLb = GL2[Ib[1]]
        cost = _task_cost(1, Int(_isidentity_B(W1, Ib)), GLb, B1b)
        t, cache = _be_task(GLb, B1b)
        push!(tasks, _CostedTask(cost, :BE, t, (; gl = Ib[1], gc = 0, cache)))
        T = promote_type(T, eltype(GLb), eltype(B1b))
    end
    T = _push_costed_boundary!(
        tasks, T, :II, _apply_II!,
        size(W2, 4) == 1 ? missing : transpose(removeunit(GR[1], 2))
    )
    T = _push_costed_boundary!(
        tasks, T, :ID, _apply_ID!,
        nonzero_length(W2.D) > 0 ? only(W2.D) : missing
    )
    T = _push_costed_boundary!(
        tasks, T, :DE, _apply_DE!,
        nonzero_length(W1.D) > 0 ? only(W1.D) : missing
    )
    T = _push_costed_boundary!(
        tasks, T, :EE, _apply_EE!,
        size(W1, 1) == 1 ? missing : removeunit(GL[end], 2)
    )
    tasks = _sorted_tasks(tasks)
    return ThreadedJordanMPO_AC2_Hamiltonian(tasks, nothing, T, _accumulator_workspace())
end

function (H::ThreadedJordanMPO_AC2_Hamiltonian)(x::MPOTensor)
    T = promote_type(eltype(x), H.promoted_eltype)
    y = vi_zerovector(x, T)
    H.plan !== nothing && return _execute_tree!(y, H.plan, H.workspace, x)
    return _execute_tasks!(y, H.tasks, H.workspace, x)
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

# ---------------------------------------------------------------------------
# Read-only diagnostics (task pool composition, cache/scratch footprints,
# executor statistics of the last matvec).
# ---------------------------------------------------------------------------

const _ThreadedHamiltonian =
    Union{ThreadedJordanMPO_AC_Hamiltonian, ThreadedJordanMPO_AC2_Hamiltonian}

"""
    task_report(H) -> NamedTuple

Read-only diagnostics of a threaded effective Hamiltonian: task counts per
kind, distinct MPO block rows used on the left (`gl`, shared by AA/AB/BE) and
on the right (`gc`, shared by CA/CB/IC), stage-cache and executor workspace
footprints, and per-worker statistics of the most recent matvec.
"""
function task_report(H::_ThreadedHamiltonian)
    H.plan === nothing && return _task_report_flat(H)
    return _tree_report(H)
end

function _task_report_flat(H::_ThreadedHamiltonian)
    tasks = H.tasks
    counts = Dict{Symbol,Int}()
    for t in tasks
        counts[t.kind] = get(counts, t.kind, 0) + 1
    end
    gls = [t.meta.gl for t in tasks if t.meta.gl > 0]
    gcs = [t.meta.gc for t in tasks if t.meta.gc > 0]
    warm = 0
    cache_bytes = 0
    for t in tasks
        c = t.meta.cache
        (c === nothing || c[] === nothing) && continue
        warm += 1
        cache_bytes += sum(Base.summarysize, c[].tensors)
    end
    ws = H.workspace
    return (;
        ntasks = length(tasks),
        kind_counts = counts,
        gl_paths = length(gls), gl_rows = length(unique(gls)),
        gc_paths = length(gcs), gc_rows = length(unique(gcs)),
        stage_cache_warm = warm,
        stage_cache_bytes = cache_bytes,
        accumulator_bytes = sum(Base.summarysize, ws.buffers; init = 0),
        scratch_bytes = sum(s -> length(s.buffer), ws.scratch; init = 0),
        scratch_highwater = sum(s -> Int(s.max_offset), ws.scratch; init = 0),
        worker_nrun = copy(ws.nrun),
        worker_merge_s = copy(ws.tmerge),
    )
end

function _tree_report(H::_ThreadedHamiltonian)
    plan = H.plan
    counts = Dict{Symbol,Int}()
    for tasks in (plan.phaseA, plan.phaseB, plan.phaseC), t in tasks
        counts[t.kind] = get(counts, t.kind, 0) + 1
    end
    warm = 0
    cache_bytes = 0
    for c in plan.caches
        c[] === nothing && continue
        warm += 1
        cache_bytes += c[] isa _StageCache ?
            sum(Base.summarysize, c[].tensors) : Base.summarysize(c[])
    end
    gcs = [t.meta.gc for ph in (plan.phaseA, plan.phaseB, plan.phaseC)
           for t in ph if t.meta.gc > 0]
    ws = H.workspace
    return (;
        ntasks = length(plan.phaseA) + length(plan.phaseB) + length(plan.phaseC),
        kind_counts = counts,
        phases = (A = length(plan.phaseA), B = length(plan.phaseB),
                  C = length(plan.phaseC)),
        gl_paths = plan.nleft, gl_rows = length(plan.rows_T),
        gc_paths = length(gcs), gc_rows = length(unique(gcs)),
        pairs_S = length(plan.pairs_S), nmid_paths = plan.nmid,
        gl_redundancy = plan.nleft / max(length(plan.rows_T), 1),
        s_redundancy = plan.nmid / max(length(plan.pairs_S), 1),
        tree_cache_warm = warm,
        tree_cache_bytes = cache_bytes,
        accumulator_bytes = sum(Base.summarysize, ws.buffers; init = 0),
        scratch_bytes = sum(s -> length(s.buffer), ws.scratch; init = 0),
        scratch_highwater = sum(s -> Int(s.max_offset), ws.scratch; init = 0),
        worker_nrun = copy(ws.nrun),
        worker_merge_s = copy(ws.tmerge),
    )
end

function Base.show(io::IO, H::_ThreadedHamiltonian)
    rep = task_report(H)
    kinds = sort!(collect(rep.kind_counts); by = last, rev = true)
    kind_str = join(("$k=$n" for (k, n) in kinds), " ")
    return print(io, typeof(H).name.name,
        "($(rep.ntasks) tasks: $kind_str; gl rows $(rep.gl_rows)/$(rep.gl_paths), gc rows $(rep.gc_rows)/$(rep.gc_paths))")
end
