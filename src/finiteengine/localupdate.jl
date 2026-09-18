# Fast local updates — Phase 2 of the finite engine.
#
# `HalfFiniteEnvironments`-dispatched methods of MPSKit's `local_update!`,
# mirroring MPSKit's own methods (algorithms/groundstate/dmrg.jl) step for
# step — same expand → eigsolve → gauge structure, same adaptive-solver
# control, same Galerkin/decay bookkeeping — with three differences:
#
#   1. the environments are lazy and self-invalidating (environments.jl);
#   2. they return the local eigenvalue `λ`, which the driver uses as the
#      per-sweep energy estimate;
#   3. optional `energy_shift`: solves `(H - E_shift)·x` and adds the shift
#      back to `λ` (a pure shift is a Krylov no-op).
#
# A plain `TruncatedAlgorithm` gauge goes through the block-parallel SVD
# (factorizations.jl) when enabled via `configure_finite_engine!(;
# svd_threaded = true)`, MPSKit's serial path otherwise. DMRG3S's noisy gauge
# is not supported on the half engine (it needs H and the environments to
# build its perturbation).

"""
    local_update!(site/pos, direction, ψ, H, alg, env::HalfFiniteEnvironments,
        ϵ_global, ϵ_trunc, decay_rate, timer, allocator;
        energy_shift = 0.0)

One-site (`alg::DMRG`) or two-site (`alg::DMRG2`) local update on the half
engine. Unlike MPSKit's own methods (which return
`(ψ, ϵ_local, ϵ_trunc, decay_rate)`), these return
`(ψ, λ, ϵ_local, ϵ_trunc, decay_rate)` with `λ` the local eigenvalue (energy
estimate of the normalized state after the update).
"""
function local_update!(
        site, direction::Val,
        ψ, O, alg::DMRG, env::HalfFiniteEnvironments,
        ϵ_global, ϵ_trunc, decay_rate,
        timeroutput, allocator;
        energy_shift::Real = 0.0
    )
    ϵ_local = calc_galerkin(site, ψ, O, ψ, env; alg.backend, allocator)

    # 1. expand (CBE): mutates AL[site]/AR[site±1]; the === staleness checks in
    #    the environment manager make the dependent environments rebuild lazily
    if !isnothing(alg.alg_expand)
        @timeit timeroutput "expand" changebond!(
            site, direction, ψ, O, alg.alg_expand, env; allocator
        )
    end

    # 2. local update
    alg_eigsolve = adapt_solver(
        alg.alg_eigsolve; decay_rate, g_local = ϵ_local, g_global = ϵ_global, eps_trunc = ϵ_trunc
    )
    ac_old = ψ.AC[site]
    λ, AC′, info = @timeit timeroutput "AC_eigsolve" begin
        H_effective = AC_hamiltonian(site, ψ, O, ψ, env; alg.backend, allocator)
        _fixedpoint_shifted(H_effective, ac_old, alg_eigsolve, energy_shift)
    end

    # 3. gauge
    ψ, ϵ_trunc = @timeit timeroutput "gauge" _fast_gauge!(
        ψ, site, direction, AC′, alg.alg_gauge; normalize = true
    )

    # 4. bookkeeping (identical to MPSKit)
    decay_rate = clamp((first(info.normres) / ϵ_local)^(1 / max(1, info.numops)), 1.0e-3, 0.999)

    return ψ, λ, ϵ_local, ϵ_trunc, decay_rate
end

function local_update!(
        pos, direction::Val,
        ψ, O, alg::DMRG2, env::HalfFiniteEnvironments,
        ϵ_global, ϵ_trunc, decay_rate,
        timeroutput, allocator;
        energy_shift::Real = 0.0
    )
    Heff = @timeit timeroutput "AC2_hamiltonian" AC2_hamiltonian(pos, ψ, O, ψ, env; alg.backend, allocator)

    kind = direction === Val(:right) ? :ACAR : :ALAC
    ac2 = AC2(ψ, pos; kind)
    AC2′ = normalize!(Heff * ac2)
    project_complement!(AC2′, ψ.AL[pos])
    ϵ_local = norm(AC2′)

    # 1. local two-site update
    alg_eigsolve = adapt_solver(
        alg.alg_eigsolve; decay_rate, g_local = ϵ_local, g_global = ϵ_global, eps_trunc = ϵ_trunc
    )
    λ, newA2center, info = @timeit timeroutput "AC2_eigsolve" begin
        _fixedpoint_shifted(Heff, ac2, alg_eigsolve, energy_shift)
    end

    # 2. gauge: truncated SVD split back into single-site tensors
    ψ, ϵ_trunc = @timeit timeroutput "gauge" _fast_gauge2!(
        ψ, pos, direction, newA2center, alg.alg_gauge
    )

    # 3. bookkeeping (identical to MPSKit)
    decay_rate = clamp((first(info.normres) / ϵ_local)^(1 / max(1, info.numops)), 1.0e-3, 0.999)

    return ψ, λ, ϵ_local, ϵ_trunc, decay_rate
end

# fixedpoint with optional spectral shift; returns (λ_true, x, info)
function _fixedpoint_shifted(H_effective, x0, alg_eigsolve, energy_shift::Real)
    iszero(energy_shift) && return fixedpoint(H_effective, x0, :SR, alg_eigsolve)
    f = x -> H_effective(x) - energy_shift * x
    λ, x, info = fixedpoint(f, x0, :SR, alg_eigsolve)
    return λ + energy_shift, x, info
end

# gauge dispatch: block-parallel truncated SVD for a plain truncated gauge
# when enabled via `configure_finite_engine!(; svd_threaded = true)`, MPSKit's
# own (operator/environments-free) `gauge!`/`gauge2!` otherwise
function _fast_gauge!(ψ, pos, direction::Val, AC, alg_gauge; normalize::Bool)
    if alg_gauge isa TruncatedAlgorithm && _SVD_THREADED[] && Threads.nthreads() > 1
        return _threaded_gauge!(ψ, pos, direction, AC, alg_gauge; normalize)
    end
    return gauge!(ψ, pos, direction, AC, alg_gauge; normalize)
end

function _fast_gauge2!(ψ, pos, direction::Val, AC2, alg_gauge)
    if alg_gauge isa TruncatedAlgorithm && _SVD_THREADED[] && Threads.nthreads() > 1
        return _threaded_gauge2!(ψ, pos, direction, AC2, alg_gauge; normalize = true)
    end
    return gauge2!(ψ, pos, direction, AC2, alg_gauge; normalize = true)
end

# ---------------------------------------------------------------------------
# environment freeing after each update (driver calls these right after
# `local_update!`); see docs/finite-engine-redesign.md for the derivation
# ---------------------------------------------------------------------------

# one-site (DMRG / TDVP1) L2R at site pos: GRs[pos+1] was consumed by the
# update and is not needed again this sweep
_free_after_move!(env::HalfFiniteEnvironments, ::Union{DMRG, TDVP}, ::Val{:right}, pos::Int) =
    free_right!(env, pos + 1)
# one-site (DMRG / TDVP1) R2L at site pos: GLs[pos] consumed
_free_after_move!(env::HalfFiniteEnvironments, ::Union{DMRG, TDVP}, ::Val{:left}, pos::Int) =
    free_left!(env, pos)
# two-site (DMRG2 / TDVP2) L2R at bond pos: GRs[pos+1] is stale (AR[pos+1]
# changed), GRs[pos+2] consumed; GRs[N+1] is the boundary and stays
function _free_after_move!(env::HalfFiniteEnvironments, ::Union{DMRG2, TDVP2}, ::Val{:right}, pos::Int)
    free_right!(env, pos + 1)
    pos + 2 <= length(env.GRs) - 1 && free_right!(env, pos + 2)
    return nothing
end
# two-site (DMRG2 / TDVP2) R2L at bond pos: GLs[pos+1] stale (AL[pos] changed),
# GLs[pos] consumed; GLs[1] is the boundary and stays
function _free_after_move!(env::HalfFiniteEnvironments, ::Union{DMRG2, TDVP2}, ::Val{:left}, pos::Int)
    free_left!(env, pos + 1)
    pos >= 2 && free_left!(env, pos)
    return nothing
end
