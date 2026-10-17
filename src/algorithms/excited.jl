# Penalized excited-state DMRG on the fast finite engine.
#
# Computes low-lying eigenstates of a finite MPO Hamiltonian in a fixed
# symmetry sector by the penalty method: the n-th state is the ground state
# of H + Σᵢ λᵢ |ψᵢ⟩⟨ψᵢ| over all previously found states ψᵢ
# (the strategy of MPSKit's `FiniteExcited`, default weight 10), but the
# local updates run on the package's own engine instead of MPSKit's
# `LazyLincoCache`:
#
#   * the Hamiltonian term lives in a disk-backable `HalfFiniteEnvironments`;
#   * each penalty term contributes the exact rank-one operator |uᵢ⟩⟨uᵢ|
#     on the local tensor, where u_i is ψᵢ's site tensor folded into the
#     ⟨ψ|ψᵢ⟩ overlap environments. This is the same action as MPSKit's
#     `ProjectionDerivativeOperator`, but built on plain two-leg overlap
#     environments (`HalfFiniteEnvironments(ψ, nothing, ψᵢ)`) instead
#     of the doubled four-leg ones - the overlap environments are tiny
#     (O(D^2) per entry) and stay in RAM;
#   * convergence: after the `truncdims` schedule is exhausted, tail sweeps
#     at the final bond dimension continue until |ΔE| ≤ etol, where E is
#     the bare ⟨ψ|H|ψ⟩ (not the penalized eigenvalue). A state is
#     therefore never passed on to the next penalized run while still
#     under-converged - the sequential error-propagation problem that
#     `FiniteExcited` leaves open. Optionally, `polish` re-sweeps every
#     computed state against the final versions of all the others (see
#     [`excited_states`](@ref)).
#
# The drivers mirror dmrg1!/dmrg2! (algorithms/dmrg.jl) and reuse their
# building blocks (fixed-budget Lanczos eigensolver, CBE expansion, gauge
# steps, environment freeing schedule, per-update diagnostics).

# ---------------------------------------------------------------------------
# folded local tensors and the penalized local operator
# ---------------------------------------------------------------------------

# uᵢ = GLᵢ · ψᵢ.AC[site] · GRᵢ : ψᵢ folded into the ⟨ψ|ψᵢ⟩
# overlap environments, shaped like ψ.AC[site] (the legs of the local
# variational tensor), so that ⟨uᵢ|x⟩ is the overlap of ψᵢ with the state
# whose site-`site` tensor is replaced by x
function _fold_ac(oenv::HalfFiniteEnvironments, site::Int, state::AbstractFiniteMPS)
    GL = leftenv(oenv, site, state)
    GR = rightenv(oenv, site, state)
    A = oenv.above.AC[site]
    return @plansor u[-1 -2; -3] := GL[-1; 1] * A[1 -2; 2] * GR[2; -3]
end

function _fold_ac2(oenv::HalfFiniteEnvironments, pos::Int, state::AbstractFiniteMPS)
    GL = leftenv(oenv, pos, state)
    GR = rightenv(oenv, pos + 1, state)
    A1 = oenv.above.AC[pos]
    A2 = oenv.above.AR[pos + 1]
    # (2,2) partition (Vl P1; Vr P2), matching MPSKit's AC2 layout
    # (numerically identical to repartition of the (3,1) fold - verified)
    return @plansor u[-1 -2; -3 -4] := GL[-1; 1] * A1[1 -2; 2] * A2[2 -4; 3] * GR[3; -3]
end

# H_eff(x) = H_AC(x) + Σᵢ λᵢ uᵢ⟨uᵢ|x⟩ - the penalty terms are
# exact rank-one operators, so the eigensolver sees them at negligible cost
struct PenalizedLocalOperator{H, U <: Tuple, W <: Tuple}
    h::H
    us::U
    weights::W
end

function (p::PenalizedLocalOperator)(x)
    y = p.h(x)
    for (u, w) in zip(p.us, p.weights)
        y += w * dot(u, x) * u
    end
    return y
end

Base.:*(p::PenalizedLocalOperator, x) = p(x)

# ---------------------------------------------------------------------------
# penalized local updates (mirror finiteengine/localupdate.jl)
# ---------------------------------------------------------------------------

function penalized_local_update!(
        site, direction::Val,
        ψ, O, alg::DMRG, henv::HalfFiniteEnvironments, oenvs::Tuple, weights::Tuple,
        timeroutput, allocator;
        energy_shift::Real = 0.0
    )
    # H-environment rebuild forced up front under its own timer section
    # (mirror of finiteengine/localupdate.jl; the overlap environments rebuild
    # lazily inside the folds below and stay attributed to the constructor)
    t_env = _force_envs!(timeroutput, henv, site, site, ψ)
    ϵ_local = calc_galerkin(site, ψ, O, ψ, henv; alg.backend, allocator)

    # 1. expand (CBE): driven by the H environments only - the penalties do
    #    not take part in the bond expansion; the overlap environments
    #    rebuild lazily from the replaced AL/AR tensors
    t_expand = 0.0
    if !isnothing(alg.alg_expand)
        t0 = time_ns()
        @timeit timeroutput "expand" changebond!(
            site, direction, ψ, O, alg.alg_expand, henv; allocator
        )
        t_expand = _elapsed_ns(t0)
    end

    # 2. penalized local update (the expand replaced tensors, so the H
    #    environments are stale again — force the second rebuild first)
    t_env += _force_envs!(timeroutput, henv, site, site, ψ)
    t0 = time_ns()
    H_ac = @timeit timeroutput "AC_hamiltonian" AC_hamiltonian(
        site, ψ, O, ψ, henv; alg.backend, allocator
    )
    us = ntuple(i -> _fold_ac(oenvs[i], site, ψ), length(oenvs))
    Heff = PenalizedLocalOperator(H_ac, us, weights)
    t_ham = _elapsed_ns(t0)
    ac_old = ψ.AC[site]
    t_mv = Ref(0.0)
    t0 = time_ns()
    λ, ACnew, info = @timeit timeroutput "AC_eigsolve" begin
        _fixedpoint_shifted(Heff, ac_old, alg.alg_eigsolve, energy_shift, timeroutput, t_mv)
    end
    t_eig = _elapsed_ns(t0)

    # 3. gauge
    t0 = time_ns()
    ψ, ϵ_trunc = @timeit timeroutput "gauge" _fast_gauge!(
        ψ, site, direction, ACnew, alg.alg_gauge; normalize = true
    )
    t_gauge = _elapsed_ns(t0)

    diag = (; numops = info.numops, t_env, t_ham, t_eig, t_mv = t_mv[], t_expand, t_gauge)
    return ψ, λ, ϵ_local, ϵ_trunc, diag
end

function penalized_local_update!(
        pos, direction::Val,
        ψ, O, alg::DMRG2, henv::HalfFiniteEnvironments, oenvs::Tuple, weights::Tuple,
        timeroutput, allocator;
        energy_shift::Real = 0.0
    )
    # H-environment rebuild forced up front under its own timer section; the
    # "AC2_hamiltonian" bucket then measures only the constructor (plus the
    # penalty folds, which lazily rebuild the overlap environments)
    t_env = _force_envs!(timeroutput, henv, pos, pos + 1, ψ)
    t0 = time_ns()
    Heff = @timeit timeroutput "AC2_hamiltonian" begin
        H_ac2 = AC2_hamiltonian(pos, ψ, O, ψ, henv; alg.backend, allocator)
        us = ntuple(i -> _fold_ac2(oenvs[i], pos, ψ), length(oenvs))
        PenalizedLocalOperator(H_ac2, us, weights)
    end
    t_ham = _elapsed_ns(t0)

    kind = direction === Val(:right) ? :ACAR : :ALAC
    ac2 = AC2(ψ, pos; kind)
    AC2new = normalize!(Heff * ac2)
    project_complement!(AC2new, ψ.AL[pos])
    ϵ_local = norm(AC2new)

    # 1. penalized local two-site update
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

# ---------------------------------------------------------------------------
# sweep driver
# ---------------------------------------------------------------------------

# per-sweep bare energy: a single-site contraction with exact environments
# (MPSKit's `dot(bra, H, ket, envs)` has an untyped `envs` argument, so the
# half engine plugs in directly)
_excited_energy(ψ::FiniteMPS, H, henv) = real(dot(ψ, H, ψ, henv)) / norm(ψ)^2

function _excited_run!(
        label::String, ψ::FiniteMPS, H,
        algs::AbstractVector{<:Union{DMRG, DMRG2}},
        henv::HalfFiniteEnvironments, oenvs::Tuple, weights::Tuple;
        etol::Float64, max_tailsweeps::Int,
        filename::String, save::Union{Bool, AbstractVector{<:Integer}}, verbose,
        manual_gc::Bool = true
    )
    N = length(ψ)
    isempty(algs) && throw(ArgumentError("truncdims cannot be empty"))
    algs = collect(algs)   # tail sweeps push! into this
    nscheduled = length(algs)
    save_iters = save isa Bool ? Int[] : collect(Int, save)
    all(>=(1), save_iters) || throw(ArgumentError("save indices must be >= 1"))
    n = maximum(alg -> alg isa DMRG ? N : N - 1, algs)
    ϵ_locals = ones(n)        # per-position Galerkin errors (per-update log; checkpoint)
    ϵ_truncs = zeros(n)       # per-position truncation errors of the gauge step
    allocator = default_allocator(ψ, SerialScheduler())
    timer = TimerOutput()
    wpos = ndigits(N)
    witer = ndigits(nscheduled)
    wD = 4
    E_prev, E₀, ΔE = NaN, NaN, NaN
    wrote = false
    start_time, record_start = now(), now()
    Int(verbose) > 0 && println(
        label, " Sweep Started (half engine, ", length(oenvs), " penalized state(s)): ",
        Dates.format(start_time, "d.u yyyy HH:MM")
    )
    Int(verbose) > 0 && flush(stdout)

    iter = 0
    while iter < length(algs)
        iter += 1
        alg = algs[iter]
        fwd, bwd = alg isa DMRG ? (1:(N - 1), N:-1:2) : (1:(N - 1), (N - 2):-1:1)
        positions = union(fwd, bwd)
        shift = !isnan(E_prev) ? E_prev : 0.0
        λ = 0.0

        gc0 = Base.gc_num()
        sweepdiag = _SweepDiag()
        @timeit timer "L2R sweep" begin
            for pos in fwd
                ψ, λ, ϵ_locals[pos], ϵ_truncs[pos], diag =
                    penalized_local_update!(
                        pos, Val(:right), ψ, H, alg, henv, oenvs, weights,
                        timer, allocator; energy_shift = shift
                    )
                sweepdiag = _accum!(sweepdiag, diag)
                _free_after_move!(henv, alg, Val(:right), pos)
                foreach(oenv -> _free_after_move!(oenv, alg, Val(:right), pos), oenvs)
                if Int(verbose) > 1
                    _dmrg_log_move(alg, Val(:right), pos, ψ, λ, ϵ_locals[pos], ϵ_truncs[pos], wpos, wD, diag)
                    flush(stdout)
                end
            end
        end
        @timeit timer "R2L sweep" begin
            for pos in bwd
                ψ, λ, ϵ_locals[pos], ϵ_truncs[pos], diag =
                    penalized_local_update!(
                        pos, Val(:left), ψ, H, alg, henv, oenvs, weights,
                        timer, allocator; energy_shift = shift
                    )
                sweepdiag = _accum!(sweepdiag, diag)
                _free_after_move!(henv, alg, Val(:left), pos)
                foreach(oenv -> _free_after_move!(oenv, alg, Val(:left), pos), oenvs)
                if Int(verbose) > 1
                    _dmrg_log_move(alg, Val(:left), pos, ψ, λ, ϵ_locals[pos], ϵ_truncs[pos], wpos, wD, diag)
                    flush(stdout)
                end
            end
        end

        # bare energy: the penalized eigenvalue λ is not the physical energy
        E₀ = @timeit(timer, "expectation_value", _excited_energy(ψ, H, henv))
        ΔE = abs(E₀ - E_prev)
        E_prev = E₀
        # penalty magnitude Σᵢ λᵢ|⟨ψᵢ|ψ⟩|² (~0 at convergence)
        P = isempty(oenvs) ? 0.0 :
            sum(w * abs2(dot(oenv.above, ψ)) for (w, oenv) in zip(weights, oenvs))
        Dmax = maximum(b -> dim(right_virtualspace(ψ, b)), 1:(N - 1))
        wD = max(wD, ndigits(Dmax))
        witer = max(witer, ndigits(iter))
        current_time = now()
        if manual_gc
            GC.gc(true)
            Int(verbose) > 0 && _print_mem("sweep $iter done", henv)
        end
        if Int(verbose) > 0
            println(
                "[", lpad(iter, witer), "] ", label, "/",
                nameof(typeof(alg)), " sweep | duration: ",
                Dates.canonicalize(current_time - start_time)
            )
            @printf(
                "  E = %.10f | D = %*d | ΔE = %.3e | P = %.3e | max ϵ = %.3e | max ϵtr = %.3e\n",
                E₀, wD, Dmax, ΔE, P,
                maximum(view(ϵ_locals, positions)), maximum(view(ϵ_truncs, positions))
            )
            _print_sweep_diag(
                stdout, sweepdiag, gc0, Base.gc_num(),
                (current_time - start_time).value / 1000
            )
            flush(stdout)
        end

        # tail sweeps at the final bond dimension until |ΔE| ≤ etol; `done`
        # marks the actual final iteration (guards the save=true checkpoint)
        done = iter == length(algs) &&
            ((ΔE <= etol) || iter - nscheduled >= max_tailsweeps)
        if iter == length(algs) && !done
            push!(algs, algs[end])
        end

        if (save === true && done) || iter in save_iters
            mode = wrote ? "a" : "w"
            jldopen(filename, mode) do f
                f["sweep_$(iter)_ψ"] = ψ
                f["sweep_$(iter)_E"] = E₀
                f["sweep_$(iter)_ΔE"] = ΔE
                f["sweep_$(iter)_ϵ"] = ϵ_locals
                f["sweep_$(iter)_ϵtrunc"] = ϵ_truncs
                f["sweep_$(iter)_D"] = Dmax
                f["sweep_$(iter)_P"] = P
            end
            wrote = true
        end
        start_time = current_time
    end

    record_end = now()
    if !(ΔE <= etol) && max_tailsweeps > 0 && iter - nscheduled >= max_tailsweeps
        @warn "$label did not reach |ΔE| ≤ etol = $etol after $max_tailsweeps tail sweep(s) (ΔE = $ΔE)"
    end
    if Int(verbose) > 0
        println(
            "Ended: ", Dates.format(record_end, "d.u yyyy HH:MM"),
            " | total duration: ", Dates.canonicalize(record_end - record_start),
            " | ", iter, " sweep(s) (", iter - nscheduled, " tail)"
        )
        for (i, oenv) in enumerate(oenvs)
            @printf("  |⟨ψᵢ[%d]|ψ⟩| = %.3e\n", i, abs(dot(oenv.above, ψ)))
        end
        println(timer)
        _print_krylov_breakdown(timer)
        flush(stdout)
    end
    return ψ, henv, E_prev
end

# ---------------------------------------------------------------------------
# public drivers
# ---------------------------------------------------------------------------

function _check_penalize_sectors(ψ::FiniteMPS, penalize)
    N = length(ψ)
    for (i, ψᵢ) in enumerate(penalize)
        length(ψᵢ) == N || throw(DimensionMismatch(
            "penalized state $i has length $(length(ψᵢ)), expected $N"
        ))
        (left_virtualspace(ψ, 1) == left_virtualspace(ψᵢ, 1) &&
         right_virtualspace(ψ, N) == right_virtualspace(ψᵢ, N)) || throw(ArgumentError(
            "penalized state $i is in a different symmetry sector (boundary " *
            "virtual spaces do not match)"
        ))
    end
    return nothing
end

_penalty_weights(weight::Real, m::Int) = ntuple(_ -> Float64(weight), m)
function _penalty_weights(weights::AbstractVector{<:Real}, m::Int)
    length(weights) == m || throw(ArgumentError(
        "need exactly one weight per penalized state (got $(length(weights)) for $m states)"
    ))
    return Tuple(Float64.(weights))
end

"""
    excited_dmrg2!(ψ, H, penalize, truncdims; kwargs...)

Two-site finite DMRG for the lowest eigenstate of `H` orthogonal to the
`penalize` states, on the fast finite engine. The state is optimized as the
ground state of `H + Σᵢ wᵢ |ψᵢ⟩⟨ψᵢ|` with penalty weight(s)
`weight` (default 10, a scalar or one value per penalized state); an empty
`penalize` runs a plain ground-state search.

After the `truncdims` schedule is exhausted, tail sweeps at the final bond
dimension continue until the per-sweep change of the bare energy
`E = ⟨ψ|H|ψ⟩` satisfies `|ΔE| ≤ etol` (default `1e-8`), at most
`max_tailsweeps` extra sweeps.

`ψ` and all `penalize` states must be `FiniteMPS` of the same length in
the same symmetry sector. The Hamiltonian-term environments are built with
the `disk` keyword as in [`dmrg2!`](@ref); the (tiny) overlap environments
stay in RAM. `save` defaults to `false`; the other keyword arguments are as
in [`dmrg2!`](@ref). Returns `(ψ, envs, E)`.
"""
function excited_dmrg2!(ψ::FiniteMPS, H, penalize::AbstractVector{<:FiniteMPS},
        truncdims::AbstractVector{<:Integer};
        weight::Union{Real, AbstractVector{<:Real}} = 10.0,
        etol::Float64 = 1.0e-8,
        max_tailsweeps::Integer = 20,
        alg_eigsolve = _default_alg_eigsolve(),
        alg_svd = SafeDivideAndConquer(),
        filename::String = "default_excited_dmrg2.jld2",
        save::Union{Bool, AbstractVector{<:Integer}} = false,
        verbose::Union{Bool, Integer} = true,
        disk::Union{Bool, AbstractString} = false,
        manual_gc::Bool = true)
    _check_penalize_sectors(ψ, penalize)
    algs = map(truncdims) do D
        DMRG2(; alg_eigsolve, alg_svd, trunc = truncrank(Int(D)))
    end
    henv = HalfFiniteEnvironments(ψ, H; disk)
    oenvs = ntuple(i -> HalfFiniteEnvironments(ψ, nothing, penalize[i]), length(penalize))
    weights = _penalty_weights(weight, length(penalize))
    return _excited_run!(
        "excited-DMRG2", ψ, H, algs, henv, oenvs, weights;
        etol = Float64(etol), max_tailsweeps = Int(max_tailsweeps),
        filename, save, verbose, manual_gc
    )
end

"""
    excited_dmrg1!(ψ, H, penalize, truncdims; kwargs...)

One-site version of [`excited_dmrg2!`](@ref) with optional bond expansion;
`alg_expand` defaults to the CBE strategy of [`dmrg1!`](@ref) and is driven
by the Hamiltonian environments only.
"""
function excited_dmrg1!(ψ::FiniteMPS, H, penalize::AbstractVector{<:FiniteMPS},
        truncdims::AbstractVector{<:Integer};
        weight::Union{Real, AbstractVector{<:Real}} = 10.0,
        etol::Float64 = 1.0e-8,
        max_tailsweeps::Integer = 20,
        alg_eigsolve = _default_alg_eigsolve(),
        alg_svd = SafeDivideAndConquer(),
        alg_expand = D -> CBEExpand(; trunc = truncrank(ceil(Int, 0.1 * D)), alg_svd),
        filename::String = "default_excited_dmrg1.jld2",
        save::Union{Bool, AbstractVector{<:Integer}} = false,
        verbose::Union{Bool, Integer} = true,
        disk::Union{Bool, AbstractString} = false,
        manual_gc::Bool = true)
    _check_penalize_sectors(ψ, penalize)
    algs = map(truncdims) do D
        DMRG(;
            alg_eigsolve, alg_svd, trunc = truncrank(Int(D)),
            alg_expand = _dmrg_expand_alg(alg_expand, Int(D))
        )
    end
    henv = HalfFiniteEnvironments(ψ, H; disk)
    oenvs = ntuple(i -> HalfFiniteEnvironments(ψ, nothing, penalize[i]), length(penalize))
    weights = _penalty_weights(weight, length(penalize))
    return _excited_run!(
        "excited-DMRG1", ψ, H, algs, henv, oenvs, weights;
        etol = Float64(etol), max_tailsweeps = Int(max_tailsweeps),
        filename, save, verbose, manual_gc
    )
end

# ---------------------------------------------------------------------------
# full low-lying-state workflow
# ---------------------------------------------------------------------------

# local reimplementation of MPSKit's internal `randomize!`: fill every block
# of the tensor with standard normal noise (any storagetype)
function _randomize!(a::AbstractTensorMap)
    for (k, v) in blocks(a)
        map!(x -> randn(typeof(x)), v, v)
    end
    return a
end

# perturbed copy of ψ (noise of relative amplitude `atol`), the same style
# of initial-state perturbation as MPSKit's `FiniteExcited`
function _perturb_state(ψ::FiniteMPS; atol::Real = 1.0e-2)
    return FiniteMPS(map(1:length(ψ)) do i
        A = copy(i == length(ψ) ? ψ.AC[i] : ψ.AL[i])
        noise = _randomize!(similar(A))
        return A + (atol / norm(noise)) * noise
    end)
end

# convergence diagnostics: energies and the overlap matrix of all states
function _excited_report(states, energies; verbose)
    Int(verbose) > 0 || return nothing
    m = length(states)
    @printf("  state %d done | E = %.12f\n", m - 1, energies[end])
    m == 1 && return nothing
    O = [abs(dot(states[i], states[j])) for i in 1:m, j in 1:m]
    maxoff = maximum(O[i, j] for i in 1:m for j in (i + 1):m)
    maxdev = maximum(abs(O[i, i] - 1) for i in 1:m)
    @printf("  overlaps: max off-diagonal = %.3e | max |⟨ψᵢ|ψᵢ⟩-1| = %.3e\n", maxoff, maxdev)
    if m <= 10
        for i in 1:m
            print("   ")
            for j in 1:m
                @printf(" %.2e", O[i, j])
            end
            print("\n")
        end
    end
    flush(stdout)
    return nothing
end

"""
    excited_states(H, init, nstates, truncdims; kwargs...)

Compute the `nstates` lowest-lying eigenstates of `H` in the symmetry sector
of `init` and return `(states, energies)` as plain vectors. State 0 is the
sector ground state (no penalty); state n is optimized with the penalized
driver against all previously found states. Every state is converged with
tail sweeps to `|ΔE| ≤ etol` before the next one starts, and the overlap
matrix is reported after each state.

# Keyword arguments
- `driver`: per-state optimizer (default [`excited_dmrg2!`](@ref);
  [`excited_dmrg1!`](@ref) is the one-site alternative)
- `weight`: penalty weight(s), default 10
- `inits`: optional vector of initial states for states `1:nstates-1`
  (`inits[n]` initializes state `n`); by default each excited run starts
  from a perturbed copy of the previous state (amplitude `perturb`, default
  1e-2)
- `polish`: after all states are found, re-converge every excited state
  against the final versions of all the others (repairs the sequential
  error propagation); polish sweeps use the final bond dimension and at
  most `polish_tailsweeps` tail sweeps
- `save`: write each converged state to `filename` (JLD2 keys `"state_n"`,
  `"energy_n"`, plus the collected `"states"`/`"energies"` at the end);
  polished states overwrite their entries
- remaining keywords (`etol`, `max_tailsweeps`, `disk`, `verbose`,
  `manual_gc`, `alg_eigsolve`, `alg_svd`, ...) are forwarded to `driver`
"""
function excited_states(H, init::FiniteMPS, nstates::Integer,
        truncdims::AbstractVector{<:Integer};
        driver = excited_dmrg2!,
        weight::Union{Real, AbstractVector{<:Real}} = 10.0,
        inits = nothing,
        perturb::Real = 1.0e-2,
        polish::Bool = false,
        polish_tailsweeps::Integer = 5,
        etol::Float64 = 1.0e-8,
        max_tailsweeps::Integer = 20,
        filename::String = "default_excited_states.jld2",
        save::Bool = true,
        verbose::Union{Bool, Integer} = true,
        kwargs...)
    nstates >= 1 || throw(ArgumentError("nstates must be >= 1"))
    states = FiniteMPS[]
    energies = Float64[]
    for n in 0:(nstates - 1)
        ψn = if n == 0
            copy(init)
        elseif !isnothing(inits)
            inits[n]
        else
            _perturb_state(states[end]; atol = perturb)
        end
        Int(verbose) > 0 && println("=== excited_states: state $n / $(nstates - 1) ===")
        flush(stdout)
        ψn, _, E = driver(
            ψn, H, states, truncdims;
            weight, etol, max_tailsweeps, verbose, save = false, kwargs...
        )
        push!(states, ψn)
        push!(energies, E)
        _excited_report(states, energies; verbose)
        save && jldopen(filename, n == 0 ? "w" : "a") do f
            f["state_$n"] = ψn
            f["energy_$n"] = E
        end
    end
    if polish && nstates > 1
        Dmax = maximum(truncdims)
        for k in 2:nstates
            others = states[[1:(k - 1); (k + 1):nstates]]
            Int(verbose) > 0 && println("=== excited_states: polish state $(k - 1) ===")
            flush(stdout)
            states[k], _, E = driver(
                states[k], H, others, [Dmax];
                weight, etol, max_tailsweeps = polish_tailsweeps, verbose,
                save = false, kwargs...
            )
            energies[k] = E
            save && jldopen(filename, "r+") do f
                f["state_$(k - 1)"] = states[k]
                f["energy_$(k - 1)"] = E
            end
        end
        Int(verbose) > 0 && println("=== excited_states: overlaps after polish ===")
        _excited_report(states, energies; verbose)
    end
    save && jldopen(filename, "r+") do f
        f["states"] = states
        f["energies"] = energies
    end
    return states, energies
end
