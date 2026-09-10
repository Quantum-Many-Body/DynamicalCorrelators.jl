# Fast local updates — Phase 2 of the finite engine.
#
# `fast_local_update!` mirrors MPSKit's `local_update!` (algorithms/groundstate/
# dmrg.jl) step for step — same expand → eigsolve → gauge structure, same
# adaptive-solver control, same Galerkin/decay bookkeeping — with three
# differences:
#
#   1. It is tied to `FastFiniteEnvironments` (the `leftenv`/`rightenv`
#      adapters give MPSKit's building blocks — `calc_galerkin`, `changebond!`,
#      `AC_hamiltonian`, `gauge!` — lazy, self-invalidating environments).
#   2. It returns the eigensolve eigenvalue `λ`, which the driver uses as the
#      per-sweep energy estimate (the λ of the last update of a sweep equals
#      the variational energy of the normalized state), avoiding MPSKit's full
#      `expectation_value` rebuild every sweep.
#   3. Optional `energy_shift`: solves `(H - E_shift)·x` instead of `H·x`
#      (KrylovKit accepts plain callables) and adds the shift back to `λ`.
#      Shifting the spectrum towards the small-magnitude end stabilizes and
#      speeds up the :SR Lanczos at large D.
#
# The post-update gauge is routed through the block-parallel truncated SVD
# (factorizations.jl) whenever the gauge is a plain
# `MatrixAlgebraKit.TruncatedAlgorithm` (i.e. DMRG2, and DMRG with `trunc` set)
# and `Threads.nthreads() > 1` — it parallelizes a stage that is otherwise
# serial, so there is nothing to conflict with. Expanding gauges (`DMRG3S`)
# and QR gauges keep MPSKit's path.

"""
    fast_local_update!(site/pos, direction, ψ, H, alg, env::FastFiniteEnvironments,
        ϵ_global, ϵ_trunc, decay_rate, iter, timer, allocator;
        energy_shift = 0.0)

One-site (`alg::DMRG`) or two-site (`alg::DMRG2`) local update on the fast
engine. Returns `(ψ, λ, ϵ_local, ϵ_trunc, decay_rate)` where `λ` is the local
eigenvalue (energy estimate of the normalized state after the update).
"""
function fast_local_update!(
        site, direction::Val,
        ψ, O, alg::DMRG, env::FastFiniteEnvironments,
        ϵ_global, ϵ_trunc, decay_rate,
        iter, timeroutput, allocator;
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

    alg_gauge = _update_alg_gauge(alg.alg_gauge, iter, ϵ_global)

    # 3. gauge
    ψ, ϵ_trunc = @timeit timeroutput "gauge" _fast_gauge!(
        ψ, site, direction, O, env, AC′, alg_gauge; alg.backend, allocator
    )

    # 4. bookkeeping (identical to MPSKit)
    decay_rate = clamp((first(info.normres) / ϵ_local)^(1 / max(1, info.numops)), 1.0e-3, 0.999)

    return ψ, λ, ϵ_local, ϵ_trunc, decay_rate
end

function fast_local_update!(
        pos, direction::Val,
        ψ, O, alg::DMRG2, env::FastFiniteEnvironments,
        ϵ_global, ϵ_trunc, decay_rate,
        iter, timeroutput, allocator;
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

    alg_gauge = _update_alg_gauge(alg.alg_gauge, iter, ϵ_global)

    # 2. gauge: truncated SVD split back into single-site tensors
    ψ, ϵ_trunc = @timeit timeroutput "gauge" _fast_gauge2!(
        ψ, pos, direction, newA2center, alg_gauge
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

# gauge dispatch: threaded block-parallel SVD for a plain truncated gauge when
# Julia threads are available, MPSKit's own gauge! otherwise (also covers QR
# gauges and DMRG3S).
function _fast_gauge!(
        ψ, pos, direction::Val, O, env, AC, alg_gauge;
        backend, allocator
    )
    if alg_gauge isa TruncatedAlgorithm && Threads.nthreads() > 1
        return _threaded_gauge!(ψ, pos, direction, AC, alg_gauge; normalize = true)
    end
    return gauge!(
        ψ, pos, direction, O, env, AC, alg_gauge; normalize = true, backend, allocator
    )
end

function _fast_gauge2!(ψ, pos, direction::Val, AC2, alg_gauge)
    if alg_gauge isa TruncatedAlgorithm && Threads.nthreads() > 1
        return _threaded_gauge2!(ψ, pos, direction, AC2, alg_gauge; normalize = true)
    end
    return gauge2!(ψ, pos, direction, AC2, alg_gauge; normalize = true)
end

# ---------------------------------------------------------------------------
# environment freeing after each update (driver calls these right after
# `fast_local_update!`); see docs/finite-engine-redesign.md for the derivation
# ---------------------------------------------------------------------------

# one-site (DMRG / TDVP1) L2R at site pos: GRs[pos+1] was consumed by the
# update and is not needed again this sweep
_free_after_move!(env::FastFiniteEnvironments, ::Union{DMRG, TDVP}, ::Val{:right}, pos::Int) =
    free_right!(env, pos + 1)
# one-site (DMRG / TDVP1) R2L at site pos: GLs[pos] consumed
_free_after_move!(env::FastFiniteEnvironments, ::Union{DMRG, TDVP}, ::Val{:left}, pos::Int) =
    free_left!(env, pos)
# two-site (DMRG2 / TDVP2) L2R at bond pos: GRs[pos+1] is stale (AR[pos+1]
# changed), GRs[pos+2] consumed; GRs[N+1] is the boundary and stays
function _free_after_move!(env::FastFiniteEnvironments, ::Union{DMRG2, TDVP2}, ::Val{:right}, pos::Int)
    free_right!(env, pos + 1)
    pos + 2 <= length(env.GRs) - 1 && free_right!(env, pos + 2)
    return nothing
end
# two-site (DMRG2 / TDVP2) R2L at bond pos: GLs[pos+1] stale (AL[pos] changed),
# GLs[pos] consumed; GLs[1] is the boundary and stays
function _free_after_move!(env::FastFiniteEnvironments, ::Union{DMRG2, TDVP2}, ::Val{:left}, pos::Int)
    free_left!(env, pos + 1)
    pos >= 2 && free_left!(env, pos)
    return nothing
end
