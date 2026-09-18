# Controlled Bond Expansion (CBE) as an MPSKit `changebond!` algorithm.
#
# `CBEExpand` selects expansion directions from the dominant singular vectors
# of the projected two-site update `g2 = NL†·H|ψ₂⟩·NR̄`, where `NL`/`NR` are
# isometries onto the orthogonal complements of the current tensors — the same
# quantity MPSKit's `OptimalExpand` uses (the "direct" selection of Gleis, Li
# & von Delft, PRL 130, 246402 (2023)). The difference is purely
# computational: `OptimalExpand` first applies the two-site effective
# Hamiltonian to the full `(D, d, d, D)` two-site state and only then projects
# onto the complement, while here every nonzero Jordan-MPO channel is
# contracted directly into the projected matrix, so the full two-site vector
# is never formed and the sparse channel structure is distributed over Julia
# threads.
#
# Only the four channel classes acting non-trivially on BOTH physical legs
# survive the nullspace projection: `AA` (continuing-continuing), `CA`
# (starting-continuing), `AB` (continuing-ending) and `CB` (starting-ending).
# Every other block of the two-site derivative acts as the identity on at
# least one physical leg and is annihilated by `NL†` or `NR̄`.
#
# Non-Jordan MPOs fall back to one explicit two-site matvec
# (`AC2_projection`), i.e. exactly `OptimalExpand`'s selection.
#
# The channel contractions are the ones validated by the pre-v0.14 CBE
# implementation (v0.13 `hamiltonian_derivatives_multithreading.jl`), written
# in the @plansor planar conventions of src/draft/tensorkit-plansor-conventions.md
# — do not "fix" leg orders without a numerical check against `AC2_projection`.

"""
    CBEExpand(; trunc, alg_svd = SafeDivideAndConquer())

Controlled Bond Expansion (direct selection) as an MPSKit `changebond!`
bond-expansion algorithm, usable everywhere `OptimalExpand` is accepted — the
`alg_expand` keyword of [`dmrg1!`](@ref)/[`dmrg_mix!`](@ref), MPSKit's `DMRG`
and `TDVP` drivers, or a standalone `changebonds!` call.

The expansion directions are the dominant singular vectors of the projected
two-site update `NL† H|ψ₂⟩ NR̄`, where `NL`/`NR` are isometries onto the
orthogonal complement of the current tensors. For Jordan-MPO Hamiltonians this
matrix is assembled channel by channel directly in the complement — the full
two-site effective-Hamiltonian application that makes `OptimalExpand`
expensive is never formed. Other MPO types fall back to one explicit two-site
matvec (identical selection to `OptimalExpand`).

`trunc` bounds how much is *added* to each bond, not the total bond dimension
that is kept (same convention as `OptimalExpand`); the gauge step of the local
update truncates the enlarged bond back to the sweep's target. The projected
block is normalized before the SVD, so only `truncrank`/`truncspace` have a
robust meaning here.

# Fields
- `trunc::TruncationStrategy`: how many directions are *added* to each bond
- `alg_svd`: SVD algorithm (default: `SafeDivideAndConquer()`)
"""
@kwdef struct CBEExpand{S} <: Algorithm
    alg_svd::S = SafeDivideAndConquer()
    trunc::TruncationStrategy
end

# ---------------------------------------------------------------------------
# channel-decomposed projected two-site update (Jordan MPOs)
# ---------------------------------------------------------------------------

"""Nonzero `C-A` (starting-continuing) channel of a two-site Jordan-MPO sandwich."""
struct _CAChannel{O1, O2, R}
    localop1::O1
    localop2::O2
    rightenv::R
end

"""Nonzero `A-B` (continuing-ending) channel of a two-site Jordan-MPO sandwich."""
struct _ABChannel{L, O1, O2}
    leftenv::L
    localop1::O1
    localop2::O2
end

"""Nonzero `C-B` (starting-ending) channel of a two-site Jordan-MPO sandwich."""
struct _CBChannel{O1, O2}
    localop1::O1
    localop2::O2
end

function _collect_CA_channels(C1, A2, GR2)
    channels = _CAChannel[]
    sizehint!(channels, nonzero_length(C1) * nonzero_length(A2))
    for (I1, W1ij) in nonzero_pairs(C1), (I2, W2jk) in nonzero_pairs(A2)
        I1[3] == I2[1] || continue
        push!(channels, _CAChannel(W1ij, W2jk, GR2[I2[4]]))
    end
    return channels
end

function _collect_AB_channels(GL2, A1, B2)
    channels = _ABChannel[]
    sizehint!(channels, nonzero_length(A1) * nonzero_length(B2))
    for (I1, W1ij) in nonzero_pairs(A1), (I2, W2jk) in nonzero_pairs(B2)
        I1[4] == I2[1] || continue
        push!(channels, _ABChannel(GL2[I1[1]], W1ij, W2jk))
    end
    return channels
end

function _collect_CB_channels(C1, B2)
    channels = _CBChannel[]
    sizehint!(channels, nonzero_length(C1) * nonzero_length(B2))
    for (I1, W1ij) in nonzero_pairs(C1), (I2, W2jk) in nonzero_pairs(B2)
        I1[3] == I2[1] || continue
        push!(channels, _CBChannel(W1ij, W2jk))
    end
    return channels
end

# Each `_cbe_apply` contracts one channel directly into the projected matrix
# `g2 = NL†·H|ψ₂⟩·NR̄`: `left` is the left site tensor (`AC[site]` for
# `Val(:right)`, `AL[site-1]` for `Val(:left)`) and `right_tail` is
# `_transpose_tail` of the right one (`AR[site+1]`/`AC[site]`). The @plansor
# leg orderings follow the planar conventions of the validated pre-v0.14 code.
function _cbe_apply(ch::_AAChannel, NL, NR, left, right_tail)
    @plansor tmp[-1; -2] :=
        conj(NL[4 5; -1]) * ch.leftenv[4 2; 1] * left[1 3; 11] * right_tail[11; 6 7] *
        ch.localop1[2 5; 3 12] * ch.localop2[12 10; 7 8] * ch.rightenv[6 8; 9] *
        conj(NR[-2; 9 10])
    return tmp
end

function _cbe_apply(ch::_CAChannel, NL, NR, left, right_tail)
    @plansor tmp[-1; -2] :=
        conj(NL[1 3; -1]) * left[1 2; 9] * right_tail[9; 4 5] * ch.localop1[3; 2 10] *
        ch.localop2[10 8; 5 6] * ch.rightenv[4 6; 7] * conj(NR[-2; 7 8])
    return tmp
end

function _cbe_apply(ch::_ABChannel, NL, NR, left, right_tail)
    @plansor tmp[-1; -2] :=
        conj(NL[4 5; -1]) * ch.leftenv[4 2; 1] * left[1 3; 9] * right_tail[9; 6 7] *
        ch.localop1[2 5; 3 10] * ch.localop2[10 8; 7] * conj(NR[-2; 6 8])
    return tmp
end

function _cbe_apply(ch::_CBChannel, NL, NR, left, right_tail)
    @plansor tmp[-1; -2] :=
        conj(NL[1 3; -1]) * left[1 2; 7] * right_tail[7; 4 5] * ch.localop1[3; 2 8] *
        ch.localop2[8 6; 5] * conj(NR[-2; 4 6])
    return tmp
end

# linear index across the heterogeneously typed channel groups
function _cbe_channel_at(groups::Tuple, idx::Int)
    for group in groups
        n = length(group)
        idx <= n && return group[idx]
        idx -= n
    end
    throw(BoundsError(groups, idx))
end

# Work-stealing reduction over the channel groups with per-task local
# accumulation and a single locked merge per task (same pattern as
# `_apply_channels` in hamiltonian_threaded.jl). Contractions inside spawned
# tasks use the default allocator: MPSKit's scratch BufferAllocator is not
# thread-safe. Returns `nothing` when no channel contributes.
function _cbe_apply_all(groups::Tuple, NL, NR, left, right_tail)
    total = sum(length, groups)
    total == 0 && return nothing

    if Threads.nthreads() == 1 || total == 1
        acc = nothing
        for group in groups, ch in group
            tmp = _cbe_apply(ch, NL, NR, left, right_tail)
            acc = acc === nothing ? tmp : acc + tmp
        end
        return acc
    end

    idx = Threads.Atomic{Int}(1)
    lk = ReentrantLock()
    total_acc = Ref{Any}(nothing)
    nt = min(Threads.nthreads(), total)
    Threads.@sync for _ in 1:nt
        Threads.@spawn begin
            local_acc = nothing
            while true
                i = Threads.atomic_add!(idx, 1)
                i > total && break
                tmp = _cbe_apply(_cbe_channel_at(groups, i), NL, NR, left, right_tail)
                local_acc = local_acc === nothing ? tmp : local_acc + tmp
            end
            if local_acc !== nothing
                lock(lk) do
                    total_acc[] = total_acc[] === nothing ? local_acc : total_acc[] + local_acc
                end
            end
        end
    end
    return total_acc[]
end

# ---------------------------------------------------------------------------
# projected two-site update g2 = NL†·H|ψ₂⟩·NR̄
# ---------------------------------------------------------------------------

# Jordan MPO: channel-decomposed direct contraction into the complement.
function _cbe_g2(
        site::Int, ::Val, ψ::AbstractFiniteMPS,
        H::MPOHamiltonian{<:JordanMPOTensor}, envs, left, right, NL, NR
    )
    GL = leftenv(envs, site, ψ)
    GR = rightenv(envs, site + 1, ψ)
    W1, W2 = H[site], H[site + 1]
    GL2, GR2 = GL[2:(end - 1)], GR[2:(end - 1)]
    groups = (
        _collect_AA_channels(GL2, W1.A, W2.A, GR2),
        _collect_CA_channels(W1.C, W2.A, GR2),
        _collect_AB_channels(GL2, W1.A, W2.B),
        _collect_CB_channels(W1.C, W2.B),
    )
    g2 = _cbe_apply_all(groups, NL, NR, left, _transpose_tail(right))
    return g2 === nothing ? zerovector!(similar(left, space(NL, 3) ← space(NR, 1))) : g2
end

# generic MPO fallback: one explicit two-site matvec, then project — the same
# selection as MPSKit's `OptimalExpand`
function _cbe_g2(
        site::Int, dir::Val, ψ::AbstractFiniteMPS, H, envs, left, right, NL, NR
    )
    kind = dir === Val(:right) ? :ACAR : :ALAC
    ac2 = AC2_projection(site, ψ, H, ψ, envs; kind)
    return adjoint(NL) * ac2 * adjoint(NR)
end

# ---------------------------------------------------------------------------
# changebond! / changebonds! interface
# ---------------------------------------------------------------------------

# The state reconstruction is identical to MPSKit's `OptimalExpand`: the
# current tensor is embedded into the enlarged space with zero weight (the
# state is preserved), and the new directions live on the tensor across the
# bond so that the next local update sees them.
function changebond!(
        site::Int, ::Val{:right}, ψ::AbstractFiniteMPS, H, alg::CBEExpand, envs;
        normalize::Bool = true, allocator = nothing
    )
    left = ψ.AC[site]
    right = ψ.AR[site + 1]
    NL = left_null(left)
    NR = right_null!(_transpose_tail(right; copy = true))

    g2 = _cbe_g2(site, Val(:right), ψ, H, envs, left, right, NL, NR)
    nrm = norm(g2)
    # nothing to expand here; normalizing a zero g2 would NaN the SVD
    nrm ≤ eps(real(scalartype(g2)))^(3 / 4) && return ψ
    _, _, Vᴴ = svd_trunc!(scale!(g2, inv(nrm)); trunc = alg.trunc, alg = alg.alg_svd)

    # optimal vectors at site+1; embed `left` with zero weight in the new
    # directions, leaving the state unchanged
    ar_re = Vᴴ * NR
    nal_space = codomain(left) ← (only(domain(left)) ⊕ space(Vᴴ, 1))
    nal, nc, _ = left_gauge(absorb!(zerovector!(similar(left, nal_space)), left))
    nar = _transpose_front(catcodomain(_transpose_tail(right), ar_re))

    normalize && normalize!(nc)
    ψ.AC[site] = (nal, nc)
    ψ.AC[site + 1] = (nc, nar)
    return ψ
end

function changebond!(
        site::Int, ::Val{:left}, ψ::AbstractFiniteMPS, H, alg::CBEExpand, envs;
        normalize::Bool = true, allocator = nothing
    )
    left = ψ.AL[site - 1]
    right = ψ.AC[site]
    NL = left_null(left)
    NR = right_null!(_transpose_tail(right; copy = true))

    g2 = _cbe_g2(site - 1, Val(:left), ψ, H, envs, left, right, NL, NR)
    nrm = norm(g2)
    nrm ≤ eps(real(scalartype(g2)))^(3 / 4) && return ψ
    U, _, _ = svd_trunc!(scale!(g2, inv(nrm)); trunc = alg.trunc, alg = alg.alg_svd)

    # optimal vectors at site-1; embed `right` with zero weight in the new
    # directions, leaving the state unchanged
    Q = NL * U
    right_tail = _transpose_tail(right)
    nc_space = (codomain(right_tail)[1] ⊕ _lastspace(Q)') ← domain(right_tail)
    nc, Qr = lq_compact!(absorb!(zerovector!(similar(right_tail, nc_space)), right_tail))
    AL_exp = catdomain(left, Q)

    normalize && normalize!(nc)
    ψ.AC[site] = (nc, _transpose_front(Qr))
    ψ.AC[site - 1] = (AL_exp, nc)
    return ψ
end

changebonds(ψ::AbstractFiniteMPS, H, alg::CBEExpand, envs = environments(ψ, H, ψ)) =
    changebonds!(copy(ψ), H, alg, envs)

function changebonds!(
        ψ::AbstractFiniteMPS, H, alg::CBEExpand, envs = environments(ψ, H, ψ)
    )
    for i in 1:(length(ψ) - 1)
        changebond!(i, Val(:right), ψ, H, alg, envs)
    end
    return ψ, envs
end
