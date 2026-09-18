# Finite-system DMRG drivers with an explicit per-sweep bond-dimension schedule.
#
# These bypass MPSKit's `find_groundstate!` and drive the sweep loop directly
# on MPSKit's lower-level building blocks (`local_update!`,
# `default_allocator`), rebuilding the cheap `DMRG`/`DMRG2` algorithm object
# with `trunc = truncrank(D)` each sweep, so `truncdims[i]` is exactly the kept
# bond dimension of sweep `i`. The default one-site bond expansion is
# [`CBEExpand`](@ref) (algorithms/cbe_expand.jl): the direct CBE selection, assembled
# channel by channel without materializing the two-site effective Hamiltonian.
# The single driver (`_dmrg_run!`) serves both
# environment backends: `HalfFiniteEnvironments` dispatches `local_update!` to
# the finite-engine method (finiteengine/localupdate.jl), and the few other
# backend differences are inline branches on the cache type.
#
# Note: `local_update!`, `default_allocator` and `SerialScheduler` are
# unexported MPSKit internals, not public API.

"""
    dmrg1!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector; kwargs...)

One-site finite DMRG with an explicit per-sweep bond-dimension schedule and
optional bond expansion (CBE).

`truncdims[i]` is the target bond dimension kept after sweep `i` (one sweep is
a left-to-right plus right-to-left pass), so the length of `truncdims` is
exactly the number of sweeps performed, e.g. `truncdims = [64, 64, 128, 512]`
runs two sweeps at D = 64, then one at 128 and one at 512.

# Keyword arguments
- `alg_eigsolve`: eigensolver for the one-site effective Hamiltonian
  (default: adaptive [`AdaptiveKrylov`]; pass an explicit `Lanczos(...)` to pin
  fixed Krylov parameters)
- `alg_svd`: SVD algorithm (default: `SafeDivideAndConquer()`)
- `alg_expand`: bond-expansion strategy — an instance (used as-is for every
  sweep), a callable `D -> alg` (rebuilt from each sweep's target `D`), or
  `nothing` to disable expansion (plain one-site DMRG, which cannot grow the
  bond). The default is the callable
  `D -> CBEExpand(; trunc = truncrank(ceil(Int, 0.1*D)), alg_svd)`, expanding
  by up to 10% of the target `D` before the gauge truncates back; MPSKit's
  `OptimalExpand`/`SketchedExpand`/`RandExpand` are drop-in alternatives
- `save`: JLD2 checkpointing (default: `true`) — `false` writes nothing,
  `true` stores only the final sweep, a vector of sweep indices
  (e.g. `save = [2, 4, 6]`) stores exactly those sweeps
- `filename`: JLD2 checkpoint file (default: `"default_dmrg1.jld2"`)
- `verbose`: `0` silent, `1` per-sweep summary, `>1` also per-move lines
  (default: `true`)
- `envs`: environment cache — the default `nothing` builds a
  [`HalfFiniteEnvironments`](@ref); pass MPSKit's `environments(ψ, H, ψ)` to
  run with the reference cache (e.g. for cross-checks). The cache type selects
  the local-update method, the per-sweep energy estimate and the memory
  bookkeeping of the sweep driver
- `disk`: environment disk backing when the driver builds the cache — `false`
  (default) keeps environments in RAM, `true` or a directory string serializes
  them to disk. Must be `false` when `envs` is supplied
- `manual_gc`: `HalfFiniteEnvironments` only (default: `true`): per-update
  `GC.gc(false)` plus a full collection and memory report after every sweep

Returns `(ψ, envs, E₀)`; see also [`dmrg1`](@ref), [`dmrg2!`](@ref).
"""
function dmrg1!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector{<:Integer};
        alg_eigsolve = _default_alg_eigsolve(true, 16),
        alg_svd = SafeDivideAndConquer(),
        alg_expand = D -> CBEExpand(; trunc = truncrank(ceil(Int, 0.1 * D)), alg_svd),
        filename::String = "default_dmrg1.jld2",
        save::Union{Bool, AbstractVector{<:Integer}} = true,
        verbose::Union{Bool, Integer} = true,
        disk::Union{Bool, AbstractString} = false,
        manual_gc::Bool = true,
        envs::Union{Nothing,FiniteEnvironments,HalfFiniteEnvironments} = nothing)
    algs = map(truncdims) do D
        DMRG(;
            alg_eigsolve, alg_svd, trunc = truncrank(Int(D)),
            alg_expand = _dmrg_expand_alg(alg_expand, Int(D))
        )
    end
    envs === nothing && (envs = HalfFiniteEnvironments(ψ, H; disk))
    return _dmrg_run!("DMRG1", ψ, H, algs, envs; filename, save, verbose, manual_gc)
end

"""
    dmrg1(ψ, H, truncdims; kwargs...)

Non-mutating version of [`dmrg1!`](@ref); works on a copy of `ψ`.
"""
dmrg1(ψ, H, truncdims; kwargs...) = dmrg1!(copy(ψ), H, truncdims; kwargs...)

"""
    dmrg2!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector; kwargs...)

Two-site finite DMRG with an explicit per-sweep bond-dimension schedule.

`truncdims[i]` is the target bond dimension kept after sweep `i` (a
left-to-right plus right-to-left pass), and `length(truncdims)` is exactly the
number of sweeps performed.

# Keyword arguments
Same as [`dmrg1!`](@ref), except that there is no `alg_expand` (the two-site
update grows the bond itself) and `filename` defaults to
`"default_dmrg2.jld2"`.

Returns `(ψ, envs, E₀)`; see also [`dmrg2`](@ref), [`dmrg1!`](@ref).
"""
function dmrg2!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector{<:Integer};
        alg_eigsolve = _default_alg_eigsolve(true, 16),
        alg_svd = SafeDivideAndConquer(),
        filename::String = "default_dmrg2.jld2",
        save::Union{Bool, AbstractVector{<:Integer}} = true,
        verbose::Union{Bool, Integer} = true,
        disk::Union{Bool, AbstractString} = false,
        manual_gc::Bool = true,
        envs::Union{Nothing,FiniteEnvironments,HalfFiniteEnvironments} = nothing)
    algs = map(truncdims) do D
        DMRG2(; alg_eigsolve, alg_svd, trunc = truncrank(Int(D)))
    end
    envs === nothing && (envs = HalfFiniteEnvironments(ψ, H; disk))
    return _dmrg_run!("DMRG2", ψ, H, algs, envs; filename, save, verbose, manual_gc)
end

"""
    dmrg2(ψ, H, truncdims; kwargs...)

Non-mutating version of [`dmrg2!`](@ref); works on a copy of `ψ`.
"""
dmrg2(ψ, H, truncdims; kwargs...) = dmrg2!(copy(ψ), H, truncdims; kwargs...)

"""
    dmrg_mix!(ψ::AbstractFiniteMPS, H, truncdims_2site, truncdims_1site; kwargs...)
    dmrg_mix!(ψ, H, truncdims; switch_D, kwargs...)

Hybrid finite DMRG driver: two-site sweeps ([`DMRG2`](@ref) engine) for the
small-D stages of the schedule, one-site sweeps with bond expansion
([`DMRG`](@ref) engine with `alg_expand`, i.e. CBE) for the large-D stages.
Two-site updates grow the bond fast where sweeps are cheap; one-site updates
are much cheaper per sweep at large D, where the CBE expansion only has to
refresh a small fraction of D per sweep.

Two schedule forms: `dmrg_mix!(ψ, H, [64, 128, 256], [512, 1024, 1024])` runs
the first vector with DMRG2 and the second with DMRG1+CBE (sweep numbering and
JLD2 checkpoints run continuously across both phases); the single-vector form
`dmrg_mix!(ψ, H, truncdims; switch_D = 256)` splits `truncdims` at the last
entry `≤ switch_D`.

# Keyword arguments
Same as [`dmrg1!`](@ref); `alg_expand` applies to the one-site phase only and
`filename` defaults to `"default_dmrg_mix.jld2"`.

Returns `(ψ, envs, E₀)`; see also [`dmrg_mix`](@ref), [`dmrg2!`](@ref),
[`dmrg1!`](@ref).
"""
function dmrg_mix!(
        ψ::AbstractFiniteMPS, H,
        truncdims_2site::AbstractVector{<:Integer},
        truncdims_1site::AbstractVector{<:Integer};
        alg_eigsolve = _default_alg_eigsolve(true, 16),
        alg_svd = SafeDivideAndConquer(),
        alg_expand = D -> CBEExpand(; trunc = truncrank(ceil(Int, 0.1 * D)), alg_svd),
        filename::String = "default_dmrg_mix.jld2",
        save::Union{Bool, AbstractVector{<:Integer}} = true,
        verbose::Union{Bool, Integer} = true,
        disk::Union{Bool, AbstractString} = false,
        manual_gc::Bool = true,
        envs::Union{Nothing,FiniteEnvironments,HalfFiniteEnvironments} = nothing)
    algs = Union{DMRG, DMRG2}[
        (DMRG2(; alg_eigsolve, alg_svd, trunc = truncrank(Int(D))) for D in truncdims_2site)...,
        (DMRG(; alg_eigsolve, alg_svd, trunc = truncrank(Int(D)),
            alg_expand = _dmrg_expand_alg(alg_expand, Int(D)))
            for D in truncdims_1site)...,
    ]
    envs === nothing && (envs = HalfFiniteEnvironments(ψ, H; disk))
    return _dmrg_run!("DMRG-mix", ψ, H, algs, envs; filename, save, verbose, manual_gc)
end

function dmrg_mix!(
        ψ::AbstractFiniteMPS, H, truncdims::AbstractVector{<:Integer};
        switch_D::Integer, kwargs...
    )
    i = findlast(D -> D <= switch_D, truncdims)
    i === nothing && return dmrg_mix!(ψ, H, eltype(truncdims)[], truncdims; kwargs...)
    return dmrg_mix!(ψ, H, truncdims[1:i], truncdims[(i + 1):end]; kwargs...)
end

"""
    dmrg_mix(ψ, H, args...; kwargs...)

Non-mutating version of [`dmrg_mix!`](@ref); works on a copy of `ψ`.
"""
dmrg_mix(ψ, H, args...; kwargs...) = dmrg_mix!(copy(ψ), H, args...; kwargs...)

# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

# Default local eigensolver: MPSKit's adaptive controller (`AdaptiveKrylov`),
# or with `adaptive = false` a fixed one-step `Lanczos` with the given
# `krylovdim`.
function _default_alg_eigsolve(adaptive::Bool, krylovdim::Integer)
    adaptive && return AdaptiveKrylov(; orth = ModifiedGramSchmidt())
    return Lanczos(;
        krylovdim = Int(krylovdim), maxiter = 1, tol = 1e-8,
        orth = ModifiedGramSchmidt(), eager = true, verbosity = 0
    )
end

# `alg_eigsolve === nothing` → built from `adaptive`/`krylovdim`
_resolve_alg_eigsolve(alg_eigsolve, adaptive, krylovdim) =
    something(alg_eigsolve, _default_alg_eigsolve(adaptive, krylovdim))

# Per-sweep expansion algorithm: instance used as-is / callable `D -> alg` /
# `nothing` disables expansion (see the `alg_expand` docstring of `dmrg1!`)
function _dmrg_expand_alg(alg_expand, D::Int)
    alg_expand === nothing && return nothing
    alg_expand isa Algorithm && return alg_expand
    alg_expand isa Function && return alg_expand(D)
    return throw(ArgumentError(
        "alg_expand must be `nothing`, an expansion algorithm instance " *
        "(e.g. `CBEExpand(; trunc = truncrank(k))`, `OptimalExpand(; trunc = " *
        "truncrank(k))`, `SketchedExpand(; trunc = ..., oversampling = ...)`), " *
        "or a callable `D -> alg`"
    ))
end

# bond index touched by a local update at `pos` moving in direction `dir`
_move_bond(::DMRG, ::Val{:right}, pos::Int) = pos
_move_bond(::DMRG, ::Val{:left}, pos::Int) = pos - 1
_move_bond(::DMRG2, ::Val{:right}, pos::Int) = pos
_move_bond(::DMRG2, ::Val{:left}, pos::Int) = pos

function _dmrg_log_move(alg, dir::Val{D}, pos::Int, ψ, ϵ_local, ϵ_trunc, wpos::Int, wD::Int) where {D}
    b = _move_bond(alg, dir, pos)
    # One-site R2L: the gauge just truncated the LEFT leg of AR[pos] (bond
    # pos-1). right_virtualspace(ψ, pos-1) would instead show the CBE-expanded
    # leg of AL[pos-1], which keeps the enlarged space until the next L2R pass
    # re-truncates it — same convention as MPSKit, but confusing in the log.
    Db = alg isa DMRG && D === :left ? dim(left_virtualspace(ψ, pos)) : dim(right_virtualspace(ψ, b))
    arrow = D === :right ? "=>" : "<="
    tag = D === :right ? "SweepL2R" : "SweepR2L"
    @printf(
        "  %s: site %*d %s site %*d | D = %*d | ϵ = %.2e | ϵtr = %.2e | %s\n",
        tag, wpos, b, arrow, wpos, b + 1, wD, Db, ϵ_local, ϵ_trunc,
        Dates.format(now(), "d.u yyyy HH:MM")
    )
    return nothing
end

function _dmrg_run!(
        label::String, ψ::AbstractFiniteMPS, H,
        algs::AbstractVector{<:Union{DMRG, DMRG2}},
        envs::Union{FiniteEnvironments, HalfFiniteEnvironments};
        filename::String, save::Union{Bool, AbstractVector{<:Integer}}, verbose,
        manual_gc::Bool = true
    )
    N = length(ψ)
    niters = length(algs)
    isempty(algs) && throw(ArgumentError("truncdims cannot be empty"))
    save_iters = if save isa Bool
        save ? [niters] : Int[]
    else
        iters = collect(Int, save)
        all(i -> 1 <= i <= niters, iters) ||
            throw(ArgumentError("save indices must be inside 1:$niters"))
        iters
    end
    # DMRG updates sites (n = N), DMRG2 updates bonds (n = N - 1); mixed
    # schedules size the bookkeeping arrays by the larger engine
    n = maximum(alg -> alg isa DMRG ? N : N - 1, algs)
    ϵ_locals = ones(n)          # per-position Galerkin errors (drive adaptive solvers)
    ϵ_truncs = zeros(n)         # per-position truncation errors of the gauge step
    decay_rates = zeros(n)      # per-position observed eigensolver contraction factors
    ϵ_global = 1.0
    allocator = default_allocator(ψ, SerialScheduler())
    timer = TimerOutput()
    wpos = ndigits(N)      # log column width: site index
    witer = ndigits(niters)  # log column width: sweep counter
    wD = 4                 # log column width: bond dimension (grows with D)

    # reference path needs a real energy for the first ΔE; the half engine's
    # per-sweep energy is the last update's λ, so its first ΔE is NaN
    E_prev = envs isa HalfFiniteEnvironments ? NaN : real(expectation_value(ψ, H, envs))
    λ = 0.0   # assigned inside the sweep loops; pre-declare so it is
              # visible after them (loop bodies are their own scope)
    start_time, record_start = now(), now()
    Int(verbose) > 0 && println(
        "$label Sweep Started (",
        envs isa HalfFiniteEnvironments ? "half engine" : "reference engine",
        "): ", Dates.format(start_time, "d.u yyyy HH:MM")
    )
    Int(verbose) > 0 && flush(stdout)

    for iter in 1:niters
        alg = algs[iter]
        # sweep ranges: DMRG sweeps sites 1:N-1 then N:-1:2, DMRG2 sweeps bonds
        # 1:N-1 then N-2:-1:1 (the last L2R bond was just updated)
        fwd, bwd = alg isa DMRG ? (1:(N - 1), N:-1:2) : (1:(N - 1), (N - 2):-1:1)
        # positions updated this sweep; statistics are masked to these, since
        # mixed drivers alternate between site-based (DMRG) and bond-based
        # (DMRG2) indexing
        positions = union(fwd, bwd)
        # spectral shift of this sweep's eigensolves: the half engine solves
        # `H - E_prev` (a pure shift leaves the Krylov subspace unchanged)
        shift = envs isa HalfFiniteEnvironments && !isnan(E_prev) ? E_prev : 0.0

        @timeit timer "L2R sweep" begin
            for pos in fwd
                ψ, λ, ϵ_locals[pos], ϵ_truncs[pos], decay_rates[pos] =
                    _dmrg_update!(
                        pos, Val(:right), ψ, H, alg, envs,
                        ϵ_global, ϵ_truncs[pos], decay_rates[pos],
                        iter, timer, allocator, shift
                    )
                envs isa HalfFiniteEnvironments && _free_after_move!(envs, alg, Val(:right), pos)
                manual_gc && GC.gc(false)
                ϵ_global = maximum(view(ϵ_locals, positions))
                if Int(verbose) > 1
                    _dmrg_log_move(alg, Val(:right), pos, ψ, ϵ_locals[pos], ϵ_truncs[pos], wpos, wD)
                    flush(stdout)
                end
            end
        end
        @timeit timer "R2L sweep" begin
            for pos in bwd
                ψ, λ, ϵ_locals[pos], ϵ_truncs[pos], decay_rates[pos] =
                    _dmrg_update!(
                        pos, Val(:left), ψ, H, alg, envs,
                        ϵ_global, ϵ_truncs[pos], decay_rates[pos],
                        iter, timer, allocator, shift
                    )
                envs isa HalfFiniteEnvironments && _free_after_move!(envs, alg, Val(:left), pos)
                manual_gc && GC.gc(false)
                ϵ_global = maximum(view(ϵ_locals, positions))
                if Int(verbose) > 1
                    _dmrg_log_move(alg, Val(:left), pos, ψ, ϵ_locals[pos], ϵ_truncs[pos], wpos, wD)
                    flush(stdout)
                end
            end
        end

        # per-sweep energy: the last update's λ (half engine) vs a full rebuild
        E₀ = envs isa HalfFiniteEnvironments ? real(λ) :
            @timeit(timer, "expectation_value", real(expectation_value(ψ, H, envs)))
        ΔE = abs(E₀ - E_prev)
        E_prev = E₀
        Dmax = maximum(b -> dim(right_virtualspace(ψ, b)), 1:(N - 1))
        wD = max(wD, ndigits(Dmax))
        current_time = now()
        if manual_gc
            GC.gc(true)
            Int(verbose) > 0 &&
                _print_mem("sweep $iter done", envs isa HalfFiniteEnvironments ? envs : nothing)
        end
        if Int(verbose) > 0
            println(
                "[", lpad(iter, witer), "/", niters, "] ", label, "/",
                nameof(typeof(alg)), " sweep | duration: ",
                Dates.canonicalize(current_time - start_time)
            )
            @printf(
                "  E₀ = %.10f | D = %*d | ΔE = %.3e | max ϵ = %.3e | max ϵtr = %.3e\n",
                E₀, wD, Dmax, ΔE,
                maximum(view(ϵ_locals, positions)), maximum(view(ϵ_truncs, positions))
            )
            flush(stdout)
        end
        if iter in save_iters
            mode = (iter == first(save_iters) ? "w" : "a")
            jldopen(filename, mode) do f
                f["sweep_$(iter)_ψ"] = ψ
                f["sweep_$(iter)_E"] = E₀
                f["sweep_$(iter)_ΔE"] = ΔE
                f["sweep_$(iter)_ϵ"] = ϵ_locals
                f["sweep_$(iter)_ϵtrunc"] = ϵ_truncs
                f["sweep_$(iter)_D"] = Dmax
            end
        end
        start_time = current_time
    end

    record_end = now()
    if Int(verbose) > 0
        println(
            "Ended: ", Dates.format(record_end, "d.u yyyy HH:MM"),
            " | total duration: ", Dates.canonicalize(record_end - record_start)
        )
        println(timer)
    end
    return ψ, envs, E_prev
end

# ---------------------------------------------------------------------------
# local-update dispatch
# ---------------------------------------------------------------------------

# The half-engine method (finiteengine/localupdate.jl) returns the local
# eigenvalue λ and takes `energy_shift`; the reference path is MPSKit's own
# `local_update!` (λ placeholder NaN, unused — its sweep energy comes from
# `expectation_value`).
function _dmrg_update!(pos, direction::Val, ψ, H, alg,
        envs::HalfFiniteEnvironments, ϵ_global, ϵ_trunc, decay_rate,
        iter, timer, allocator, energy_shift::Real)
    return local_update!(
        pos, direction, ψ, H, alg, envs, ϵ_global, ϵ_trunc, decay_rate,
        timer, allocator; energy_shift
    )
end
function _dmrg_update!(pos, direction::Val, ψ, H, alg,
        envs::FiniteEnvironments, ϵ_global, ϵ_trunc, decay_rate,
        iter, timer, allocator, ::Real)
    ψ, ϵ_local, ϵ_trunc, decay_rate = local_update!(
        pos, direction, ψ, H, alg, envs, ϵ_global, ϵ_trunc, decay_rate,
        iter, timer, allocator
    )
    return ψ, NaN, ϵ_local, ϵ_trunc, decay_rate
end
