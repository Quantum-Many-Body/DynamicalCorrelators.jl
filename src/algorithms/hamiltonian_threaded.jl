# Threaded channel-split evaluation of the sparse continuing blocks of Jordan-MPO
# effective Hamiltonians.
#
# MPSKit's default path prepares `GL ⊗ W.A` into a single fused (dense) tensor so
# that each matvec is one big contraction. For Hamiltonians built from sparse
# Jordan MPOs with symmetry sectors, that fusion multiplies a large number of
# exactly-zero blocks. Here the continuing `A`/`AA` blocks are instead kept as
# their nonzero channel list `(GLᵢ, Wᵢⱼ, GRⱼ)` and the channels are contracted
# independently across Julia threads (each task accumulates a local partial sum,
# merged under a lock), which exploits the sparsity *and* the available Julia
# threads. All other (dense) blocks are assembled by MPSKit's own constructors
# and applied exactly as in MPSKit.
#
# The `AC_hamiltonian`/`AC2_hamiltonian` methods overloaded here accept MPSKit's
# `prepare`/`backend`/`allocator` keywords so internal MPSKit calls (which always
# pass them) keep working unchanged. Threading is only used when
# `Threads.nthreads() > 1`, more than one channel exists, and the module toggle
# is on (see `set_threaded_hamiltonian!`); otherwise MPSKit's default operator is
# returned. Note that contractions inside spawned tasks intentionally use the
# default allocator: MPSKit's scratch BufferAllocator is not thread-safe.

"""
    set_threaded_hamiltonian!(enable::Bool) -> Bool

Enable or disable the threaded channel-split effective Hamiltonians for finite
Jordan-MPO operators (default: enabled). When disabled — or when Julia runs with
a single thread — `AC_hamiltonian`/`AC2_hamiltonian` fall back to MPSKit's
default (prepared, fused) operators. Returns the new setting.
"""
set_threaded_hamiltonian!(enable::Bool) = (_THREADED_HAMILTONIAN[] = enable)

const _THREADED_HAMILTONIAN = Ref{Bool}(true)

_threading_enabled() = _THREADED_HAMILTONIAN[] && Threads.nthreads() > 1

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

function _apply_A_channel(ch::_AChannel, x)
    @plansor tmp[-1 -2; -3] :=
        ch.leftenv[-1 5; 4] * x[4 2; 1] * ch.localop[5 -2; 2 3] * ch.rightenv[1 3; -3]
    return tmp
end

function _apply_AA_channel(ch::_AAChannel, x)
    @plansor tmp[-1 -2; -3 -4] :=
        ch.leftenv[-1 2; 1] * x[1 3; 7 5] * ch.localop1[2 -2; 3 4] *
        ch.localop2[4 -4; 5 6] * ch.rightenv[7 6; -3]
    return tmp
end

# Work-stealing over channels with per-task local accumulation and a single
# locked reduction per task.
function _apply_channels(channels, x, apply_channel)
    isempty(channels) && return zerovector(x)
    if Threads.nthreads() == 1 || length(channels) == 1
        acc = nothing
        for ch in channels
            tmp = apply_channel(ch, x)
            acc = acc === nothing ? tmp : acc + tmp
        end
        return something(acc, zerovector(x))
    end

    idx = Threads.Atomic{Int}(1)
    lk = ReentrantLock()
    total = Ref{Any}(nothing)
    nt = min(Threads.nthreads(), length(channels))
    Threads.@sync for _ in 1:nt
        Threads.@spawn begin
            local_acc = nothing
            while true
                i = Threads.atomic_add!(idx, 1)
                i > length(channels) && break
                tmp = apply_channel(channels[i], x)
                local_acc = local_acc === nothing ? tmp : local_acc + tmp
            end
            if local_acc !== nothing
                lock(lk) do
                    total[] = total[] === nothing ? local_acc : total[] + local_acc
                end
            end
        end
    end
    return something(total[], zerovector(x))
end

# ---------------------------------------------------------------------------
# threaded effective Hamiltonians
# ---------------------------------------------------------------------------

"""
One-site effective Hamiltonian of a finite Jordan-MPO operator whose sparse
continuing `A` block is evaluated as independent channels across Julia threads.
Assembled from MPSKit's unprepared `JordanMPO_AC_Hamiltonian`.
"""
struct ThreadedJordanMPO_AC_Hamiltonian{O1, O2, C} <: DerivativeOperator
    D::Union{O1, Missing}  # onsite
    I::Union{O1, Missing}  # not started
    E::Union{O1, Missing}  # finished
    C::Union{O2, Missing}  # starting
    B::Union{O2, Missing}  # ending
    channels::C            # continuing A channels
end

function ThreadedJordanMPO_AC_Hamiltonian(H0::JordanMPO_AC_Hamiltonian)
    channels = ismissing(H0.A) ? _AChannel[] :
        _collect_A_channels(H0.A.leftenv, H0.A.operators[1], H0.A.rightenv)
    return ThreadedJordanMPO_AC_Hamiltonian(H0.D, H0.I, H0.E, H0.C, H0.B, channels)
end

function (H::ThreadedJordanMPO_AC_Hamiltonian)(x::MPSTensor)
    y = _apply_channels(H.channels, x, _apply_A_channel)
    ismissing(H.D) || @plansor y[-1 -2; -3] += x[-1 1; -3] * H.D[-2; 1]
    ismissing(H.E) || @plansor y[-1 -2; -3] += H.E[-1; 1] * x[1 -2; -3]
    ismissing(H.I) || @plansor y[-1 -2; -3] += x[-1 -2; 1] * H.I[1; -3]
    ismissing(H.C) || @plansor y[-1 -2; -3] += x[-1 2; 1] * H.C[-2 -3; 2 1]
    ismissing(H.B) || @plansor y[-1 -2; -3] += H.B[-1 -2; 1 2] * x[1 2; -3]
    return y
end

"""
Two-site effective Hamiltonian of a finite Jordan-MPO operator whose sparse
continuing-continuing `AA` block is evaluated channel by channel across Julia
threads. Assembled from MPSKit's unprepared `JordanMPO_AC2_Hamiltonian`.
"""
struct ThreadedJordanMPO_AC2_Hamiltonian{O1, O2, O3, C} <: DerivativeOperator
    II::Union{O1, Missing}
    IC::Union{O2, Missing}
    ID::Union{O1, Missing}
    CB::Union{O2, Missing}
    CA::Union{O3, Missing}
    AB::Union{O3, Missing}
    channels::C
    BE::Union{O2, Missing}
    DE::Union{O1, Missing}
    EE::Union{O1, Missing}
end

function ThreadedJordanMPO_AC2_Hamiltonian(H0::JordanMPO_AC2_Hamiltonian)
    channels = ismissing(H0.AA) ? _AAChannel[] :
        _collect_AA_channels(H0.AA.leftenv, H0.AA.operators[1], H0.AA.operators[2], H0.AA.rightenv)
    return ThreadedJordanMPO_AC2_Hamiltonian(
        H0.II, H0.IC, H0.ID, H0.CB, H0.CA, H0.AB, channels, H0.BE, H0.DE, H0.EE
    )
end

function (H::ThreadedJordanMPO_AC2_Hamiltonian)(x::MPOTensor)
    y = _apply_channels(H.channels, x, _apply_AA_channel)
    ismissing(H.II) || @plansor y[-1 -2; -3 -4] += x[-1 -2; 1 -4] * H.II[-3; 1]
    ismissing(H.IC) || @plansor y[-1 -2; -3 -4] += x[-1 -2; 1 2] * H.IC[-4 -3; 2 1]
    ismissing(H.ID) || @plansor y[-1 -2; -3 -4] += x[-1 -2; -3 1] * H.ID[-4; 1]
    ismissing(H.CB) || @plansor y[-1 -2; -3 -4] += x[-1 1; -3 2] * H.CB[-2 -4; 1 2]
    ismissing(H.CA) || @plansor y[-1 -2; -3 -4] += x[-1 1; 3 2] * H.CA[-2 -4 -3; 1 2 3]
    ismissing(H.AB) || @plansor y[-1 -2; -3 -4] += x[1 2; -3 3] * H.AB[-1 -2 -4; 1 2 3]
    ismissing(H.BE) || @plansor y[-1 -2; -3 -4] += x[1 2; -3 -4] * H.BE[-1 -2; 1 2]
    ismissing(H.DE) || @plansor y[-1 -2; -3 -4] += x[-1 1; -3 -4] * H.DE[-2; 1]
    ismissing(H.EE) || @plansor y[-1 -2; -3 -4] += x[1 -2; -3 -4] * H.EE[-1; 1]
    return y
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
