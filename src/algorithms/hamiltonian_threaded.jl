# Threaded channel-split evaluation of the sparse continuing blocks of Jordan-MPO
# effective Hamiltonians.
#
# MPSKit's default path prepares `GL ⊗ W.A` into a single fused (dense) tensor so
# that each matvec is one big contraction. For Hamiltonians built from sparse
# Jordan MPOs with symmetry sectors, that fusion multiplies a large number of
# exactly-zero blocks. Here the continuing `A`/`AA` blocks are instead kept as
# their nonzero channel list `(GLᵢ, Wᵢⱼ, GRⱼ)` and contracted independently, which
# exploits the sparsity *and* the available Julia threads. The remaining dense
# boundary blocks (onsite/not-started/finished/starting/ending) enter the same
# work-stealing pool as cheap tasks, so one matvec is a single flat list of
# independent tasks with no serial tail (the old layout — channels parallel, then
# ~9 boundary terms serial — was the Amdahl bottleneck at small thread counts).
#
# Allocation discipline (what makes the channel split competitive with the fused
# operator; the structure mirrors FiniteMPS.jl's `action!`):
#   * the output `y` is allocated exactly once per matvec, with the promoted
#     eltype over the state and every operator component;
#   * every task writes its contribution *directly* into an accumulator with
#     `y += ...` — in the serial pool that accumulator is `y` itself, in the
#     threaded pool each worker owns one local accumulator (allocated once per
#     matvec) that is added into `y` under a single locked merge per worker.
#     There are no per-task output tensors and no `acc + tmp` merge copies;
#   * all contractions pass `allocator = ManualAllocator()`, so intermediate
#     (and permutation) temporaries are backed by `Libc.malloc`/`free` and
#     bypass the GC entirely. This is safe inside spawned tasks: the manual
#     allocator only backs the `istemp` intermediates (freed by the
#     @plansor-generated code itself), while accumulators stay GC-managed.
#     MPSKit's scratch BufferAllocator remains unusable here — it is not
#     thread-safe.
#
# Contraction order: left to the TensorOperations optree — at YC-Hubbard sizes
# it matches the hand-picked left-leaning (env·x first) order, while the
# right-leaning tree is ~6x slower (aa_matvec_bench.jl measurement).
#
# Numerics: identical up to floating-point reduction order (the accumulation
# order over tasks differs from the serial fused path), i.e. ~ulp-level
# differences, as with any parallel reduction.
#
# The `AC_hamiltonian`/`AC2_hamiltonian` methods overloaded here accept MPSKit's
# `prepare`/`backend`/`allocator` keywords so internal MPSKit calls (which always
# pass them) keep working unchanged. The channel-split operator is used whenever
# more than one channel exists and the module cap is nonzero (see
# `set_threaded_hamiltonian!`); the cap only controls how many worker tasks the
# pool fans out over — with cap 1 the tasks are evaluated serially in the
# calling task (the sparsity win is kept either way), and cap 0 falls back to
# MPSKit's default fused operator.
#
# BEFORE writing any new contraction here, read
# docs/tensorkit-plansor-conventions.md (the @plansor planar leg-order rules)
# and confirm the index scheme with the user first.

"""
    set_threaded_hamiltonian!(n::Integer) -> Int

Set the maximum number of worker tasks that the channel-split effective
Hamiltonians fan each matvec out over (default: `Threads.nthreads()`).
`n = 1` evaluates the tasks serially in the calling task — the sparse
channel decomposition (and its savings over MPSKit's fused dense operator) is
kept, only the threading is off; this is the right setting when the caller
already parallelizes over states. `n = 0` disables the channel-split operator
entirely, falling back to MPSKit's default fused operator. Returns the new
cap.

    set_threaded_hamiltonian!(enable::Bool) -> Int

Legacy toggle: `false` is `n = 0`, `true` is `n = Threads.nthreads()`.
"""
function set_threaded_hamiltonian!(n::Integer)
    n >= 0 || throw(ArgumentError("threaded-hamiltonian cap must be >= 0, got $n"))
    return _THREADED_HAMILTONIAN_THREADS[] = Int(n)
end
set_threaded_hamiltonian!(enable::Bool) =
    set_threaded_hamiltonian!(enable ? Threads.nthreads() : 0)

"""Current fan-out cap of the channel-split Hamiltonians (0 = disabled)."""
threaded_hamiltonian_threads() =
    _THREADED_HAMILTONIAN_THREADS[] < 0 ? Threads.nthreads() :
        _THREADED_HAMILTONIAN_THREADS[]

# -1 = uninitialized → default `Threads.nthreads()`. A lazy sentinel is needed
# because a `const` initialized with `Threads.nthreads()` at module top level
# would bake in the thread count of the *precompilation* process (usually 1)
# instead of the session's
const _THREADED_HAMILTONIAN_THREADS = Ref{Int}(-1)

"""
    set_prefused_hamiltonian!(enable::Bool) -> Bool

Enable or disable prefused one-site channels (default: disabled). When enabled,
each continuing `A` channel `(GLᵢ, Wᵢⱼ, GRⱼ)` of the threaded effective
Hamiltonian is partially contracted once up front, `P = GLᵢ ⊗ Wᵢⱼ`, so every
Krylov matvec saves re-contracting the environment with the MPO tensor. This
trades a bigger per-channel tensor for a cheaper matvec; worthwhile when the
effective Hamiltonian is applied many times per local update. Only the one-site
(AC) operator is prefused; the two-site operator is left as an extension point.
Returns the new setting.
"""
set_prefused_hamiltonian!(enable::Bool) = (_PREFUSED_HAMILTONIAN[] = enable)

const _PREFUSED_HAMILTONIAN = Ref{Bool}(false)

_threading_enabled() = threaded_hamiltonian_threads() > 0

# ---------------------------------------------------------------------------
# channels
# ---------------------------------------------------------------------------

"""Nonzero continuing `A`-block channel of a one-site effective Hamiltonian."""
struct _AChannel{L, O, R}
    leftenv::L
    localop::O
    rightenv::R
end

"""Nonzero continuing-continuing `AA`-block channel of a two-site effective Hamiltonian."""
struct _AAChannel{L, O1, O2, R}
    leftenv::L
    localop1::O1
    localop2::O2
    rightenv::R
end

function _collect_A_channels(GL2, WA, GR2)
    return map(collect(nonzero_pairs(WA))) do (I, Wij)
        _AChannel(GL2[I[1]], Wij, GR2[I[4]])
    end
end

function _collect_AA_channels(GL2, A1, A2, GR2)
    channels = _AAChannel[]
    sizehint!(channels, nonzero_length(A1) * nonzero_length(A2))
    for (I1, W1ij) in nonzero_pairs(A1), (I2, W2jk) in nonzero_pairs(A2)
        I1[4] == I2[1] || continue
        push!(channels, _AAChannel(GL2[I1[1]], W1ij, W2jk, GR2[I2[4]]))
    end
    return channels
end

# In-place channel kernels: `y += channel·x`. Temporaries (including any output
# permutation the optree needs) go through the manual allocator, so a task
# performs no GC-tracked allocation at all.
function _apply_A_channel!(y, ch::_AChannel, x)
    @plansor allocator = ManualAllocator() y[-1 -2; -3] +=
        ch.leftenv[-1 5; 4] * x[4 2; 1] * ch.localop[5 -2; 2 3] * ch.rightenv[1 3; -3]
    return y
end

function _apply_AA_channel!(y, ch::_AAChannel, x)
    @plansor allocator = ManualAllocator() y[-1 -2; -3 -4] +=
        ch.leftenv[-1 2; 1] * x[1 3; 7 5] * ch.localop1[2 -2; 3 4] *
        ch.localop2[4 -4; 5 6] * ch.rightenv[7 6; -3]
    return y
end

"""One-site channel with the environment prefused into the MPO tensor: `P = GLᵢ ⊗ Wᵢⱼ`."""
struct _PrefusedAChannel{P, R}
    fused::P
    rightenv::R
end

function _prefuse_A_channel(ch::_AChannel)
    # leg order exactly as MPSKit's `prepare_operator!!` (mpo_derivatives.jl:224):
    # P cod = (env virtual, physical bra, MPO right virtual), dom = (env virtual, physical ket)
    @plansor P[-1 -2 -3; -4 -5] := ch.leftenv[-1 5; -4] * ch.localop[5 -2; -5 -3]
    return _PrefusedAChannel(P, ch.rightenv)
end

function _apply_A_channel!(y, ch::_PrefusedAChannel, x)
    # inverse substitution of the fusion: contracts P's domain with x and P's
    # MPO-virtual codomain leg with the right environment (planar)
    @plansor allocator = ManualAllocator() y[-1 -2; -3] +=
        ch.fused[-1 -2 3; 4 2] * x[4 2; 1] * ch.rightenv[1 3; -3]
    return y
end

# ---------------------------------------------------------------------------
# boundary blocks as pool tasks
# ---------------------------------------------------------------------------

# In-place two-tensor boundary maps `y += op·x` (the permutation temporaries
# some of them need go through the manual allocator as well); pooled with the
# channels via `(y, x) -> ...` closures.
_apply_onsite!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3] += x[-1 1; -3] * op[-2; 1]; y)
_apply_finished!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3] += op[-1; 1] * x[1 -2; -3]; y)
_apply_notstarted!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3] += x[-1 -2; 1] * op[1; -3]; y)
_apply_starting!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3] += x[-1 2; 1] * op[-2 -3; 2 1]; y)
_apply_ending!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3] += op[-1 -2; 1 2] * x[1 2; -3]; y)

_apply_II!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] += x[-1 -2; 1 -4] * op[-3; 1]; y)
_apply_IC!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] += x[-1 -2; 1 2] * op[-4 -3; 2 1]; y)
_apply_ID!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] += x[-1 -2; -3 1] * op[-4; 1]; y)
_apply_CB!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] += x[-1 1; -3 2] * op[-2 -4; 1 2]; y)
_apply_CA!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] += x[-1 1; 3 2] * op[-2 -4 -3; 1 2 3]; y)
_apply_AB!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] += x[1 2; -3 3] * op[-1 -2 -4; 1 2 3]; y)
_apply_BE!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] += x[1 2; -3 -4] * op[-1 -2; 1 2]; y)
_apply_DE!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] += x[-1 1; -3 -4] * op[-2; 1]; y)
_apply_EE!(y, op, x) =
    (@plansor allocator = ManualAllocator() y[-1 -2; -3 -4] += x[1 -2; -3 -4] * op[-1; 1]; y)

function _push_boundary!(tasks, T, apply!, op)
    ismissing(op) && return T
    push!(tasks, (y, x) -> apply!(y, op, x))
    return promote_type(T, eltype(op))
end

_channel_eltype(T, ch::_AChannel) =
    promote_type(T, eltype(ch.leftenv), eltype(ch.localop), eltype(ch.rightenv))
_channel_eltype(T, ch::_AAChannel) =
    promote_type(T, eltype(ch.leftenv), eltype(ch.localop1), eltype(ch.localop2),
                 eltype(ch.rightenv))
_channel_eltype(T, ch::_PrefusedAChannel) =
    promote_type(T, eltype(ch.fused), eltype(ch.rightenv))

# Work-stealing over channels + boundary tasks. Serial pool (cap 1): every task
# accumulates directly into `y`. Threaded pool: each worker owns one local
# accumulator (a single allocation per matvec), tasks write into it, and the
# accumulators are merged into `y` under one locked in-place add per worker —
# so a matvec's GC-tracked allocations are `y` plus one accumulator per worker,
# nothing more. The clamp to >= 1 keeps an operator built while the cap was
# positive correct if the cap is lowered to 0 afterwards.
function _execute_tasks!(y, channels, apply_channel!, btasks, x)
    nch = length(channels)
    total = nch + length(btasks)
    total == 0 && return y
    nt = clamp(min(threaded_hamiltonian_threads(), Threads.nthreads()), 1, total)
    if nt == 1
        for ch in channels
            apply_channel!(y, ch, x)
        end
        for f in btasks
            f(y, x)
        end
        return y
    end
    idx = Threads.Atomic{Int}(1)
    lk = ReentrantLock()
    Threads.@sync for _ in 1:nt
        Threads.@spawn begin
            yl = VectorInterface.zerovector(y)
            while true
                i = Threads.atomic_add!(idx, 1)
                i > total && break
                i <= nch ? apply_channel!(yl, channels[i], x) : btasks[i - nch](yl, x)
            end
            lock(lk) do
                VectorInterface.add!(y, yl, VectorInterface.One(), VectorInterface.One())
            end
        end
    end
    return y
end

# ---------------------------------------------------------------------------
# threaded effective Hamiltonians
# ---------------------------------------------------------------------------

"""
One-site effective Hamiltonian of a finite Jordan-MPO operator whose sparse
continuing `A` block is evaluated as independent channels across Julia threads,
with the dense boundary blocks pooled as cheap tasks. Assembled from MPSKit's
unprepared `JordanMPO_AC_Hamiltonian`. `channels` is kept as a typed field for
inspection/probes; the matvec runs through the unified task pool.
"""
struct ThreadedJordanMPO_AC_Hamiltonian{C} <: DerivativeOperator
    channels::C               # continuing A channels
    btasks::Vector{Function}  # boundary blocks as `(y, x) -> y += ...` tasks
    promoted_eltype::Type     # promote_type over all operator components
end

function ThreadedJordanMPO_AC_Hamiltonian(H0::JordanMPO_AC_Hamiltonian)
    channels = ismissing(H0.A) ? _AChannel[] :
        _collect_A_channels(H0.A.leftenv, H0.A.operators[1], H0.A.rightenv)
    if _PREFUSED_HAMILTONIAN[] && !isempty(channels)
        channels = map(_prefuse_A_channel, channels)
    end
    btasks = Function[]
    T = Union{}
    for ch in channels
        T = _channel_eltype(T, ch)
    end
    T = _push_boundary!(btasks, T, _apply_onsite!, H0.D)
    T = _push_boundary!(btasks, T, _apply_finished!, H0.E)
    T = _push_boundary!(btasks, T, _apply_notstarted!, H0.I)
    T = _push_boundary!(btasks, T, _apply_starting!, H0.C)
    T = _push_boundary!(btasks, T, _apply_ending!, H0.B)
    return ThreadedJordanMPO_AC_Hamiltonian(channels, btasks, T)
end

function (H::ThreadedJordanMPO_AC_Hamiltonian)(x::MPSTensor)
    y = VectorInterface.zerovector(x, promote_type(eltype(x), H.promoted_eltype))
    return _execute_tasks!(y, H.channels, _apply_A_channel!, H.btasks, x)
end

"""
Two-site effective Hamiltonian of a finite Jordan-MPO operator whose sparse
continuing-continuing `AA` block is evaluated channel by channel across Julia
threads, with the dense boundary blocks pooled as cheap tasks. Assembled from
MPSKit's unprepared `JordanMPO_AC2_Hamiltonian`.
"""
struct ThreadedJordanMPO_AC2_Hamiltonian{C} <: DerivativeOperator
    channels::C               # continuing AA channels
    btasks::Vector{Function}  # boundary blocks as `(y, x) -> y += ...` tasks
    promoted_eltype::Type
end

function ThreadedJordanMPO_AC2_Hamiltonian(H0::JordanMPO_AC2_Hamiltonian)
    channels = ismissing(H0.AA) ? _AAChannel[] :
        _collect_AA_channels(H0.AA.leftenv, H0.AA.operators[1], H0.AA.operators[2],
                             H0.AA.rightenv)
    btasks = Function[]
    sizehint!(btasks, 9)
    T = Union{}
    for ch in channels
        T = _channel_eltype(T, ch)
    end
    T = _push_boundary!(btasks, T, _apply_II!, H0.II)
    T = _push_boundary!(btasks, T, _apply_IC!, H0.IC)
    T = _push_boundary!(btasks, T, _apply_ID!, H0.ID)
    T = _push_boundary!(btasks, T, _apply_CB!, H0.CB)
    T = _push_boundary!(btasks, T, _apply_CA!, H0.CA)
    T = _push_boundary!(btasks, T, _apply_AB!, H0.AB)
    T = _push_boundary!(btasks, T, _apply_BE!, H0.BE)
    T = _push_boundary!(btasks, T, _apply_DE!, H0.DE)
    T = _push_boundary!(btasks, T, _apply_EE!, H0.EE)
    return ThreadedJordanMPO_AC2_Hamiltonian(channels, btasks, T)
end

function (H::ThreadedJordanMPO_AC2_Hamiltonian)(x::MPOTensor)
    y = VectorInterface.zerovector(x, promote_type(eltype(x), H.promoted_eltype))
    return _execute_tasks!(y, H.channels, _apply_AA_channel!, H.btasks, x)
end

# ---------------------------------------------------------------------------
# AC_hamiltonian / AC2_hamiltonian overloads
# ---------------------------------------------------------------------------

# MPSKit-internal fallback: build MPSKit's own operator (optionally prepared).
function _mpskit_AC_hamiltonian(site, below, operator, envs, prepare, backend, allocator)
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site, below)
    H0 = JordanMPO_AC_Hamiltonian(GL, operator[site], GR; backend, allocator)
    return prepare ? prepare_operator!!(H0) : H0
end

function _mpskit_AC2_hamiltonian(site, below, operator, envs, prepare, backend, allocator)
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site + 1, below)
    H0 = JordanMPO_AC2_Hamiltonian(GL, operator[site], operator[site + 1], GR; backend, allocator)
    return prepare ? prepare_operator!!(H0) : H0
end

"""
    AC_hamiltonian(site, below, operator::MPOHamiltonian{<:JordanMPOTensor}, above, envs; ...)

Finite Jordan-MPO specialization that evaluates the sparse continuing `A` block
channel by channel across Julia threads. Falls back to MPSKit's default operator
when threading is disabled or pointless (single thread or at most one channel).
"""
function AC_hamiltonian(
        site::Int, below::FiniteMPS{<:MPSTensor}, operator::MPOHamiltonian{<:JordanMPOTensor},
        above::FiniteMPS{<:MPSTensor}, envs;
        prepare::Bool = true, backend::AbstractBackend = DefaultBackend(),
        allocator = DefaultAllocator()
    )
    @assert below === above "JordanMPO assumptions break"
    H0 = _mpskit_AC_hamiltonian(site, below, operator, envs, false, backend, allocator)
    if _threading_enabled() && !ismissing(H0.A) && nonzero_length(H0.A.operators[1]) > 1
        return ThreadedJordanMPO_AC_Hamiltonian(H0)
    end
    return prepare ? prepare_operator!!(H0) : H0
end

"""
    AC2_hamiltonian(site, below, operator::MPOHamiltonian{<:JordanMPOTensor}, above, envs; ...)

Finite Jordan-MPO specialization that evaluates the sparse continuing-continuing
`AA` block channel by channel across Julia threads, with the same fallback rules
as the one-site specialization.
"""
function AC2_hamiltonian(
        site::Int, below::FiniteMPS{<:MPSTensor}, operator::MPOHamiltonian{<:JordanMPOTensor},
        above::FiniteMPS{<:MPSTensor}, envs;
        prepare::Bool = true, backend::AbstractBackend = DefaultBackend(),
        allocator = DefaultAllocator()
    )
    @assert below === above "JordanMPO assumptions break"
    H0 = _mpskit_AC2_hamiltonian(site, below, operator, envs, false, backend, allocator)
    if _threading_enabled() && !ismissing(H0.AA) &&
            nonzero_length(H0.AA.operators[1]) * nonzero_length(H0.AA.operators[2]) > 1
        return ThreadedJordanMPO_AC2_Hamiltonian(H0)
    end
    return prepare ? prepare_operator!!(H0) : H0
end
