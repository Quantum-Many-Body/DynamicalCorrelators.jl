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

# Each kernel contracts one nonzero MPO path directly into the projected
# matrix `g2 = NL†·H|ψ₂⟩·NR̄`: `left` is the left site tensor (`AC[site]` for
# `Val(:right)`, `AL[site-1]` for `Val(:left)`) and `right_tail` is
# `_transpose_tail` of the right one (`AR[site+1]`/`AC[site]`). The @plansor
# leg orderings follow the planar conventions of the validated pre-v0.14 code.
# Keep `conj(...)` inline: TensorKit's planar parser relies on it to determine
# leg directions, so precomputing a conjugated tensor changes the planarity.
function _cbe_apply_aa!(y, buffer, NL, NR, left, right_tail, leftenv, localop1, localop2,
                        rightenv)
    @plansor allocator = buffer y[-1; -2] +=
        conj(NL[4 5; -1]) * leftenv[4 2; 1] * left[1 3; 11] * right_tail[11; 6 7] *
        localop1[2 5; 3 12] * localop2[12 10; 7 8] * rightenv[6 8; 9] *
        conj(NR[-2; 9 10])
    return y
end

function _cbe_apply_ca!(y, buffer, NL, NR, left, right_tail, localop1, localop2, rightenv)
    @plansor allocator = buffer y[-1; -2] +=
        conj(NL[1 3; -1]) * left[1 2; 9] * right_tail[9; 4 5] * localop1[3; 2 10] *
        localop2[10 8; 5 6] * rightenv[4 6; 7] * conj(NR[-2; 7 8])
    return y
end

function _cbe_apply_ab!(y, buffer, NL, NR, left, right_tail, leftenv, localop1, localop2)
    @plansor allocator = buffer y[-1; -2] +=
        conj(NL[4 5; -1]) * leftenv[4 2; 1] * left[1 3; 9] * right_tail[9; 6 7] *
        localop1[2 5; 3 10] * localop2[10 8; 7] * conj(NR[-2; 6 8])
    return y
end

function _cbe_apply_cb!(y, buffer, NL, NR, left, right_tail, localop1, localop2)
    @plansor allocator = buffer y[-1; -2] +=
        conj(NL[1 3; -1]) * left[1 2; 7] * right_tail[7; 4 5] * localop1[3; 2 8] *
        localop2[8 6; 5] * conj(NR[-2; 4 6])
    return y
end

# Directly collect every nonzero MPO path into one flat costed-task pool.
function _cbe_tasks(GL2, W1, W2, GR2, NL, NR, left, right_tail)
    tasks = _CostedTask[]
    T = promote_type(eltype(NL), eltype(NR), eltype(left), eltype(right_tail))

    # continuing left - continuing right
    for (I1, A1b) in nonzero_pairs(W1.A), (I2, A2b) in nonzero_pairs(W2.A)
        I1[4] == I2[1] || continue
        GLb, GRb = GL2[I1[1]], GR2[I2[4]]
        push!(tasks, _CostedTask(12, :AA, let buffer = _task_buffer()
              (y, _) -> _cbe_apply_aa!(
                  y, buffer, NL, NR, left, right_tail, GLb, A1b, A2b, GRb
              )
        end))
        T = promote_type(T, eltype(GLb), eltype(A1b), eltype(A2b), eltype(GRb))
    end
    # starting left - continuing right
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ia, A2b) in nonzero_pairs(W2.A)
        Ic[3] == Ia[1] || continue
        GRb = GR2[Ia[4]]
        push!(tasks, _CostedTask(10, :CA, let buffer = _task_buffer()
              (y, _) -> _cbe_apply_ca!(
                  y, buffer, NL, NR, left, right_tail, C1b, A2b, GRb
              )
        end))
        T = promote_type(T, eltype(C1b), eltype(A2b), eltype(GRb))
    end
    # continuing left - ending right
    for (Ia, A1b) in nonzero_pairs(W1.A), (Ib, B2b) in nonzero_pairs(W2.B)
        Ia[4] == Ib[1] || continue
        GLb = GL2[Ia[1]]
        push!(tasks, _CostedTask(10, :AB, let buffer = _task_buffer()
              (y, _) -> _cbe_apply_ab!(
                  y, buffer, NL, NR, left, right_tail, GLb, A1b, B2b
              )
        end))
        T = promote_type(T, eltype(GLb), eltype(A1b), eltype(B2b))
    end
    # starting left - ending right
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ib, B2b) in nonzero_pairs(W2.B)
        Ic[3] == Ib[1] || continue
        push!(tasks, _CostedTask(8, :CB, let buffer = _task_buffer()
              (y, _) -> _cbe_apply_cb!(
                  y, buffer, NL, NR, left, right_tail, C1b, B2b
              )
        end))
        T = promote_type(T, eltype(C1b), eltype(B2b))
    end
    return tasks, T
end

# Accumulate all paths into `y`; parallel execution uses one accumulator per
# worker and merges those buffers through `_execute_tasks!`.
function _cbe_apply_all!(y, tasks::Vector{_CostedTask})
    isempty(tasks) && return y
    btasks, _ = _sorted_tasks(tasks)
    _execute_tasks!(y, btasks, _accumulator_workspace(), nothing)
    return y
end

# ---------------------------------------------------------------------------
# projected two-site update g2 = NL†·H|ψ₂⟩·NR̄
# ---------------------------------------------------------------------------

# Jordan MPO: channel-decomposed direct contraction into the complement. The
# channel contractions below assume one physical leg per site; states with
# extra physical legs (e.g. `FiniteSuperMPS`, physical + ancilla) use the
# explicit-matvec fallback instead.
function _cbe_g2(
        site::Int, dir::Val, ψ::AbstractFiniteMPS,
        H::MPOHamiltonian{<:JordanMPOTensor}, envs, left, right, NL, NR
    )
    numout(left) == 2 || return _cbe_g2_matvec(site, dir, ψ, H, envs, NL, NR)
    GL = leftenv(envs, site, ψ)
    GR = rightenv(envs, site + 1, ψ)
    W1, W2 = H[site], H[site + 1]
    GL2, GR2 = GL[2:(end - 1)], GR[2:(end - 1)]
    right_tail = _transpose_tail(right)
    tasks, T = _cbe_tasks(GL2, W1, W2, GR2, NL, NR, left, right_tail)
    # `TensorMap{T}(undef, space)` dualizes the first argument internally;
    # pass the dual would-be codomain so the resulting tensor has the same
    # space as the channel contractions.
    g2_space = dual(space(NL, 3)) ← space(NR, 1)
    g2 = vi_zerovector(TensorMap{T}(undef, g2_space), T)
    return _cbe_apply_all!(g2, tasks)
end

# generic MPO (or multi-physical-leg state) fallback: one explicit two-site
# matvec, then project — the same selection as MPSKit's `OptimalExpand`
function _cbe_g2_matvec(site::Int, dir::Val, ψ::AbstractFiniteMPS, H, envs, NL, NR)
    kind = dir === Val(:right) ? :ACAR : :ALAC
    ac2 = AC2_projection(site, ψ, H, ψ, envs; kind)
    return adjoint(NL) * ac2 * adjoint(NR)
end

_cbe_g2(site::Int, dir::Val, ψ::AbstractFiniteMPS, H, envs, left, right, NL, NR) =
    _cbe_g2_matvec(site, dir, ψ, H, envs, NL, NR)

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
