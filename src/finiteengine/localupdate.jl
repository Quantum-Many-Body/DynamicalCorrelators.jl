# Fast local updates — Phase 2 of the finite engine.
#
# `HalfFiniteEnvironments`-dispatched methods of MPSKit's `local_update!`,
# mirroring MPSKit's own methods (algorithms/groundstate/dmrg.jl) step for
# step — same expand → eigsolve → gauge structure — with these differences:
#
#   1. the environments are lazy and self-invalidating (environments.jl);
#   2. they force the lazy environment rebuild under a dedicated
#      "env_transfer" timer section (`t_env` in the diagnostics), so the
#      Hamiltonian-construction bucket measures only the constructor;
#   3. they return the local eigenvalue `λ`, which the driver uses as the
#      per-sweep energy estimate;
#   4. optional `energy_shift`: solves `(H - E_shift)·x` and adds the shift
#      back to `λ` (a pure shift is a Krylov no-op);
#   5. the eigensolver is used as given (a concrete KrylovKit solver) — the
#      per-update `diag` NamedTuple carries the matvec count and per-phase
#      wall times (`t_env` = environment rebuild, `t_ham` = operator
#      construction, `t_eig`/`t_mv` = eigsolve and its matvec share,
#      `t_expand`, `t_gauge`) for the driver's per-update log line.
#
# A plain `TruncatedAlgorithm` gauge goes through the block-parallel SVD
# (factorizations.jl) when enabled via `configure_finite_engine!(;
# svd_threaded = true)`, MPSKit's serial path otherwise. DMRG3S's noisy gauge
# is not supported on the half engine (it needs H and the environments to
# build its perturbation).

"""
    local_update!(site/pos, direction, ψ, H, alg, env::HalfFiniteEnvironments,
        timer, allocator; energy_shift = 0.0)

One-site (`alg::DMRG`) or two-site (`alg::DMRG2`) local update on the half
engine. Unlike MPSKit's own methods (which return
`(ψ, ϵ_local, ϵ_trunc, decay_rate)`), these return
`(ψ, λ, ϵ_local, ϵ_trunc, diag)` with `λ` the local eigenvalue (energy
estimate of the normalized state after the update) and `diag` a NamedTuple
of per-update diagnostics (`numops`, `t_env`, `t_ham`, `t_eig`, `t_mv`,
`t_expand`, `t_gauge`; times in nanoseconds).
"""
function local_update!(
        site, direction::Val,
        ψ, O, alg::DMRG, env::HalfFiniteEnvironments,
        timeroutput, allocator;
        energy_shift::Real = 0.0
    )
    # environment rebuild for this update's envs, forced up front so it is
    # timed separately (calc_galerkin would otherwise trigger it untimed)
    t_env = _force_envs!(timeroutput, env, site, site, ψ)
    ϵ_local = calc_galerkin(site, ψ, O, ψ, env; alg.backend, allocator)

    # 1. expand (CBE): mutates AL[site]/AR[site±1]; the === staleness checks in
    #    the environment manager make the dependent environments rebuild lazily
    t_expand = 0.0
    if !isnothing(alg.alg_expand)
        t0 = time_ns()
        @timeit timeroutput "expand" changebond!(
            site, direction, ψ, O, alg.alg_expand, env; allocator
        )
        t_expand = _elapsed_ns(t0)
    end

    # 2. local update: the expand step replaced tensors, so the environments
    #    are stale again — force the second rebuild (still "env_transfer"),
    #    then build the effective Hamiltonian (pure constructor time) and solve
    t_env += _force_envs!(timeroutput, env, site, site, ψ)
    t0 = time_ns()
    H_effective = @timeit timeroutput "AC_hamiltonian" AC_hamiltonian(
        site, ψ, O, ψ, env; alg.backend, allocator
    )
    t_ham = _elapsed_ns(t0)
    ac_old = ψ.AC[site]
    t_mv = Ref(0.0)
    t0 = time_ns()
    λ, AC′, info = @timeit timeroutput "AC_eigsolve" begin
        _fixedpoint_shifted(H_effective, ac_old, alg.alg_eigsolve, energy_shift, timeroutput, t_mv)
    end
    t_eig = _elapsed_ns(t0)

    # 3. gauge
    t0 = time_ns()
    ψ, ϵ_trunc = @timeit timeroutput "gauge" _fast_gauge!(
        ψ, site, direction, AC′, alg.alg_gauge; normalize = true
    )
    t_gauge = _elapsed_ns(t0)

    diag = (; numops = info.numops, t_env, t_ham, t_eig, t_mv = t_mv[], t_expand, t_gauge)
    return ψ, λ, ϵ_local, ϵ_trunc, diag
end

function local_update!(
        pos, direction::Val,
        ψ, O, alg::DMRG2, env::HalfFiniteEnvironments,
        timeroutput, allocator;
        energy_shift::Real = 0.0
    )
    # environment rebuild forced up front (timed separately); the
    # "AC2_hamiltonian" bucket then measures only the operator constructor
    t_env = _force_envs!(timeroutput, env, pos, pos + 1, ψ)
    t0 = time_ns()
    Heff = @timeit timeroutput "AC2_hamiltonian" AC2_hamiltonian(pos, ψ, O, ψ, env; alg.backend, allocator)
    t_ham = _elapsed_ns(t0)

    kind = direction === Val(:right) ? :ACAR : :ALAC
    ac2 = AC2(ψ, pos; kind)
    AC2′ = normalize!(Heff * ac2)
    project_complement!(AC2′, ψ.AL[pos])
    ϵ_local = norm(AC2′)

    # 1. local two-site update
    t_mv = Ref(0.0)
    t0 = time_ns()
    λ, newA2center, info = @timeit timeroutput "AC2_eigsolve" begin
        _fixedpoint_shifted(Heff, ac2, alg.alg_eigsolve, energy_shift, timeroutput, t_mv)
    end
    t_eig = _elapsed_ns(t0)

    # 2. gauge: truncated SVD split back into single-site tensors
    t0 = time_ns()
    ψ, ϵ_trunc = @timeit timeroutput "gauge" _fast_gauge2!(
        ψ, pos, direction, newA2center, alg.alg_gauge
    )
    t_gauge = _elapsed_ns(t0)

    diag = (; numops = info.numops, t_env, t_ham, t_eig, t_mv = t_mv[], t_expand = 0.0, t_gauge)
    return ψ, λ, ϵ_local, ϵ_trunc, diag
end

# fixedpoint with optional spectral shift; returns (λ_true, x, info). Every
# application of the effective Hamiltonian is timed as the nested "matvec"
# section of the driver timer (inside the caller's "AC(2)_eigsolve" section),
# so the driver can report matvec time separately and derive the
# Gram–Schmidt/normalization/bookkeeping share as eigsolve − matvec (see
# `_print_krylov_breakdown` in algorithms/dmrg.jl) — the vector kernels
# themselves carry no instrumentation. `t_mv` additionally accumulates the
# per-update matvec wall time for the driver's per-update diagnostics.
_elapsed_ns(t0::UInt) = Float64(time_ns() - t0)

# Trigger the lazy rebuild of the two environments the update will query —
# MPSKit's constructors call `leftenv(envs, site, below)` /
# `rightenv(envs, site, below)` (one-site) or `rightenv(envs, site + 1,
# below)` (two-site) — timed under the driver's "env_transfer" section, so
# the rebuild is no longer conflated with the operator construction (the
# TimerOutput "AC(2)_hamiltonian" bucket and `t_ham` then measure only the
# constructor). A no-op when the environments are already fresh. Returns the
# wall time in nanoseconds.
function _force_envs!(timeroutput, env::HalfFiniteEnvironments, siteL::Int, siteR::Int, ψ)
    t0 = time_ns()
    @timeit timeroutput "env_transfer" begin
        leftenv(env, siteL, ψ)
        rightenv(env, siteR, ψ)
    end
    return _elapsed_ns(t0)
end

function _fixedpoint_shifted(H_effective, x0, alg_eigsolve, energy_shift::Real, timeroutput,
        t_mv::Ref{Float64} = Ref(0.0))
    f = if iszero(energy_shift)
        function (x)
            t = time_ns()
            y = @timeit(timeroutput, "matvec", H_effective(x))
            t_mv[] += _elapsed_ns(t)
            return y
        end
    else
        function (x)
            t = time_ns()
            y = @timeit(timeroutput, "matvec", H_effective(x))
            t_mv[] += _elapsed_ns(t)
            return y - energy_shift * x
        end
    end
    λ, x, info = fixedpoint(f, x0, :SR, alg_eigsolve)
    return iszero(energy_shift) ? (λ, x, info) : (λ + energy_shift, x, info)
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
