# Fast local updates — Phase 2 of the finite engine.
#
# `HalfFiniteEnvironments`-dispatched methods of MPSKit's `local_update!`,
# mirroring MPSKit's own methods (algorithms/groundstate/dmrg.jl) step for
# step — same expand → eigsolve → gauge structure — with these differences:
#
#   1. the environments are lazy and self-invalidating (environments.jl);
#   2. environments rebuild inside the Hamiltonian constructors; `t_env` is
#      kept at zero and the rebuild time is part of `t_ham`;
#   3. they return the local eigenvalue `λ`, which the driver uses as the
#      per-sweep energy estimate;
#   4. optional `energy_shift`: solves `(H - E_shift)·x` and adds the shift
#      back to `λ` (a pure shift is a Krylov no-op);
#   5. the eigensolver is used as given (a concrete KrylovKit solver) — the
#      per-update `diag` NamedTuple carries the matvec count and per-phase
#      wall times (`t_env` = zero on the lazy path, `t_ham` = operator
#      construction including any lazy environment rebuild,
#      `t_eig`/`t_mv` = eigsolve and its matvec share,
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
engine. Unlike MPSKit's own methods, these return
`(ψ, λ, ϵ_trunc, diag)` with `λ` the local eigenvalue (energy
estimate of the normalized state after the update) and `diag` a NamedTuple
of per-update diagnostics (`numops`, `t_env`, `t_ham`, `t_eig`, `t_mv`,
`t_expand`, `t_gauge`; times in nanoseconds). `t_env` is zero on the lazy
20823-style environment path; environment rebuild time is included in
`t_ham`.
"""
function local_update!(
        site, direction::Val,
        ψ, O, alg::DMRG, env::HalfFiniteEnvironments,
        timeroutput, allocator;
        energy_shift::Real = 0.0
    )
    # Lazy environments rebuild inside the Hamiltonian constructor, so that
    # time is included in `ham` rather than in a separate `env` bucket.
    t_env = 0.0

    # 1. expand (CBE): mutates AL[site]/AR[site±1]; the === staleness checks in
    #    the environment manager make the dependent environments rebuild lazily
    t_expand = 0.0
    gc_expand = 0.0
    alloc_expand = 0.0
    t_expand_can = 0.0
    t_expand_g2 = 0.0
    t_expand_svd = 0.0
    t_expand_rec = 0.0
    if !isnothing(alg.alg_expand)
        sub0 = _expand_subtimes(timeroutput)
        g0 = Base.gc_num()
        t0 = time_ns()
        @timeit timeroutput "expand" changebond!(
            site, direction, ψ, O, alg.alg_expand, env; allocator, timeroutput
        )
        t_expand = _elapsed_ns(t0)
        g1 = Base.gc_num()
        gc_expand = Float64(g1.total_time - g0.total_time)
        alloc_expand = Float64(g1.total_allocd - g0.total_allocd)
        sub1 = _expand_subtimes(timeroutput)
        t_expand_can = sub1[1] - sub0[1]
        t_expand_g2 = sub1[2] - sub0[2]
        t_expand_svd = sub1[3] - sub0[3]
        t_expand_rec = sub1[4] - sub0[4]
    end

    # 2. local update: the expand step replaced tensors, so the environments
    #    are stale again; the constructor below rebuilds them lazily.
    g0 = Base.gc_num()
    t0 = time_ns()
    H_effective = @timeit timeroutput "AC_hamiltonian" AC_hamiltonian(
        site, ψ, O, ψ, env; alg.backend, allocator
    )
    t_ham = _elapsed_ns(t0)
    g1 = Base.gc_num()
    gc_ham = Float64(g1.total_time - g0.total_time)
    alloc_ham = Float64(g1.total_allocd - g0.total_allocd)
    try
        ac_old = ψ.AC[site]
        t_mv = Ref(0.0)
        gc_mv = Ref(0.0)
        alloc_mv = Ref(0.0)
        g0 = Base.gc_num()
        t0 = time_ns()
        λ, AC′, info = @timeit timeroutput "AC_eigsolve" begin
            _fixedpoint_shifted(H_effective, ac_old, alg.alg_eigsolve, energy_shift,
                                timeroutput, t_mv, gc_mv, alloc_mv)
        end
        t_eig = _elapsed_ns(t0)
        g1 = Base.gc_num()
        gc_eig = Float64(g1.total_time - g0.total_time)
        alloc_eig = Float64(g1.total_allocd - g0.total_allocd)

        # 3. gauge
        g0 = Base.gc_num()
        t0 = time_ns()
        ψ, ϵ_trunc = @timeit timeroutput "gauge" _fast_gauge!(
            ψ, site, direction, AC′, alg.alg_gauge; normalize = true
        )
        t_gauge = _elapsed_ns(t0)
        g1 = Base.gc_num()
        gc_gauge = Float64(g1.total_time - g0.total_time)
        alloc_gauge = Float64(g1.total_allocd - g0.total_allocd)

        diag = (; numops = info.numops, t_env, t_ham, t_eig, t_mv = t_mv[], t_expand,
                t_gauge, gc_expand, gc_ham, gc_eig, gc_gauge,
                alloc_expand, alloc_ham, alloc_eig, alloc_gauge,
                t_expand_can, t_expand_g2, t_expand_svd, t_expand_rec,
                gc_mv = gc_mv[], alloc_mv = alloc_mv[],
                gc_krylov = gc_eig - gc_mv[], alloc_krylov = alloc_eig - alloc_mv[])
        return ψ, λ, ϵ_trunc, diag
    finally
        _release_hamiltonian!(H_effective)
    end
end

function local_update!(
        pos, direction::Val,
        ψ, O, alg::DMRG2, env::HalfFiniteEnvironments,
        timeroutput, allocator;
        energy_shift::Real = 0.0
    )
    # 20823-style lazy environments: the `AC2_hamiltonian` constructor
    # triggers the rebuild itself, so its time is included in `ham`.
    t_env = 0.0
    g0 = Base.gc_num()
    t0 = time_ns()
    Heff = @timeit timeroutput "AC2_hamiltonian" AC2_hamiltonian(pos, ψ, O, ψ, env; alg.backend, allocator)
    t_ham = _elapsed_ns(t0)
    g1 = Base.gc_num()
    gc_ham = Float64(g1.total_time - g0.total_time)
    alloc_ham = Float64(g1.total_allocd - g0.total_allocd)

    try
        kind = direction === Val(:right) ? :ACAR : :ALAC
        ac2 = AC2(ψ, pos; kind)

        # 1. local two-site update
        t_mv = Ref(0.0)
        gc_mv = Ref(0.0)
        alloc_mv = Ref(0.0)
        g0 = Base.gc_num()
        t0 = time_ns()
        λ, newA2center, info = @timeit timeroutput "AC2_eigsolve" begin
            _fixedpoint_shifted(Heff, ac2, alg.alg_eigsolve, energy_shift,
                                timeroutput, t_mv, gc_mv, alloc_mv)
        end
        t_eig = _elapsed_ns(t0)
        g1 = Base.gc_num()
        gc_eig = Float64(g1.total_time - g0.total_time)
        alloc_eig = Float64(g1.total_allocd - g0.total_allocd)

        # 2. gauge: truncated SVD split back into single-site tensors
        g0 = Base.gc_num()
        t0 = time_ns()
        ψ, ϵ_trunc = @timeit timeroutput "gauge" _fast_gauge2!(
            ψ, pos, direction, newA2center, alg.alg_gauge
        )
        t_gauge = _elapsed_ns(t0)
        g1 = Base.gc_num()
        gc_gauge = Float64(g1.total_time - g0.total_time)
        alloc_gauge = Float64(g1.total_allocd - g0.total_allocd)

        diag = (; numops = info.numops, t_env, t_ham, t_eig, t_mv = t_mv[],
                t_expand = 0.0, t_gauge, gc_ham, gc_eig, gc_gauge,
                gc_expand = 0.0, alloc_expand = 0.0,
                t_expand_can = 0.0, t_expand_g2 = 0.0,
                t_expand_svd = 0.0, t_expand_rec = 0.0,
                alloc_ham, alloc_eig, alloc_gauge,
                gc_mv = gc_mv[], alloc_mv = alloc_mv[],
                gc_krylov = gc_eig - gc_mv[], alloc_krylov = alloc_eig - alloc_mv[])
        return ψ, λ, ϵ_trunc, diag
    finally
        _release_hamiltonian!(Heff)
    end
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

# Cumulative wall times of the changebond! sub-phases ("canonicalize"/"g2"/
# "svd"/"reconstruct"). The "expand" section is nested under the driver's
# sweep sections, so the names are summed over the whole timer tree rather
# than looked up at a fixed depth; the names are unique to changebond!.
# A per-update share is the difference around the call (0 when never ran).
# `node` is a TimerOutput or a Section (newer TimerOutputs); both support the
# `inner_timers` compat view and `timer_time`.
function _accumulate_subtimes!(acc::Dict{String,Float64}, node)
    for (name, sec) in node.inner_timers
        acc[name] = get(acc, name, 0.0) + Float64(timer_time(sec))
        _accumulate_subtimes!(acc, sec)
    end
    return acc
end

function _expand_subtimes(timeroutput::TimerOutput)
    acc = _accumulate_subtimes!(Dict{String,Float64}(), timeroutput)
    t(name) = get(acc, name, 0.0)
    return (t("canonicalize"), t("g2"), t("svd"), t("reconstruct"))
end

function _fixedpoint_shifted(H_effective, x0, alg_eigsolve, energy_shift::Real, timeroutput,
        t_mv::Ref{Float64} = Ref(0.0), gc_mv::Ref{Float64} = Ref(0.0),
        alloc_mv::Ref{Float64} = Ref(0.0))
    f = if iszero(energy_shift)
        function (x)
            g0 = Base.gc_num()
            t = time_ns()
            y = @timeit(timeroutput, "matvec", H_effective(x))
            t_mv[] += _elapsed_ns(t)
            g1 = Base.gc_num()
            gc_mv[] += Float64(g1.total_time - g0.total_time)
            alloc_mv[] += Float64(g1.total_allocd - g0.total_allocd)
            return y
        end
    else
        function (x)
            g0 = Base.gc_num()
            t = time_ns()
            y = @timeit(timeroutput, "matvec", H_effective(x))
            t_mv[] += _elapsed_ns(t)
            g1 = Base.gc_num()
            gc_mv[] += Float64(g1.total_time - g0.total_time)
            alloc_mv[] += Float64(g1.total_allocd - g0.total_allocd)
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
