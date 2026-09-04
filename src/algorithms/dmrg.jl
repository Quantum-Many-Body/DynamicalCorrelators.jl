# Finite-system DMRG drivers with an explicit per-sweep bond-dimension schedule.
#
# These functions intentionally bypass MPSKit's `find_groundstate!` /
# `find_groundstate_sweep!` and instead drive the sweep loop themselves on top
# of the lower-level building blocks that MPSKit's own drivers use:
# `MPSKit.local_update!` (bond expansion + local eigensolve + gauge move),
# `MPSKit._sweep_ranges` / `MPSKit._num_updates` (sweep geometry), and
# `MPSKit.default_allocator` (scratch-space management). The per-sweep
# truncation target is controlled by rebuilding the (cheap) `DMRG`/`DMRG2`
# algorithm object with `trunc = truncrank(D)` at every iteration, so
# `truncdims[i]` is exactly the kept bond dimension of sweep `i`.
#
# Note: `local_update!`, `_sweep_ranges`, `_num_updates`, `default_allocator`
# and `SerialScheduler` are unexported MPSKit internals; they are the same
# functions `find_groundstate_sweep!` calls, but their signatures are not part
# of MPSKit's public API.

"""
    dmrg1!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector; kwargs...)

One-site finite DMRG with an explicit per-sweep bond-dimension schedule and
optional bond expansion (CBE).

`truncdims[i]` is the target bond dimension kept after sweep `i` (one sweep is
a left-to-right plus right-to-left pass), so the length of `truncdims` is
exactly the number of sweeps performed, e.g. `truncdims = [64, 64, 128, 512]`
runs two sweeps at D = 64, then one at 128 and one at 512. The schedule is
enforced per sweep by rebuilding MPSKit's `DMRG` algorithm object with
`trunc = truncrank(D)`; the sweep itself is driven by MPSKit's
`local_update!`, i.e. the same per-site update used by `find_groundstate!`.

# Keyword arguments
- `alg_eigsolve`: eigensolver for the one-site effective Hamiltonian
  (default: adaptive [`AdaptiveKrylov`]; pass an explicit `Lanczos(...)` to pin
  fixed Krylov parameters)
- `alg_svd`: SVD algorithm (default: `LAPACK_DivideAndConquer()`)
- `alg_expand`: bond-expansion strategy. The default `OptimalExpand` (the
  *type*) builds a fresh `OptimalExpand(; trunc = truncrank(ceil(Int, delta*D)),
  alg_svd)` for every sweep, adding up to `ceil(Int, delta*D)` directions ahead
  of the eigensolve before the gauge truncates back to `D`. Pass an *instance*
  such as `OptimalExpand(; trunc = truncrank(k))`, `SketchedExpand(;
  trunc = ...)` or `RandExpand(; trunc = ...)` to use it as-is for every sweep,
  or a callable `D -> alg` to rebuild the expansion algorithm from each sweep's
  target `D`. Pass `nothing` for plain single-site DMRG without expansion
  (which cannot grow the bond).
- `delta`: overexpansion factor used by the default `OptimalExpand` path
  (default: `0.1`)
- `filename`: JLD2 checkpoint file (default: `"default_dmrg1.jld2"`)
- `save_iters`: which sweeps to save (default: all)
- `verbose`: `0` silent, `1` per-sweep summary, `>1` also per-move lines
  (default: `true`)
- `envs`: environment cache (default: `environments(ψ, H, ψ)`)

Returns `(ψ, envs, E₀)`; see also [`dmrg1`](@ref), [`dmrg2!`](@ref).
"""
function dmrg1!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector{<:Integer};
        alg_eigsolve = _default_alg_eigsolve(true, 16),
        alg_svd = LAPACK_DivideAndConquer(),
        alg_expand = OptimalExpand,
        delta::Real = 0.1,
        filename::String = "default_dmrg1.jld2",
        save_iters::AbstractVector{<:Integer} = collect(eachindex(truncdims)),
        verbose::Union{Bool, Integer} = true,
        envs = environments(ψ, H, ψ))
    delta >= 0 || throw(ArgumentError("delta must be nonnegative"))
    algs = map(truncdims) do D
        DMRG(;
            alg_eigsolve, alg_svd, trunc = truncrank(Int(D)),
            alg_expand = _dmrg_expand_alg(alg_expand, Int(D), delta, alg_svd)
        )
    end
    return _dmrg_run!("DMRG1", ψ, H, algs, envs; filename, save_iters, verbose)
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
number of sweeps performed. The schedule is enforced per sweep by rebuilding
MPSKit's `DMRG2` algorithm object with `trunc = truncrank(D)`; each two-site
update is MPSKit's `local_update!`.

# Keyword arguments
- `alg_eigsolve`: eigensolver for the two-site effective Hamiltonian
  (default: adaptive [`AdaptiveKrylov`]; pass an explicit `Lanczos(...)` to pin
  fixed Krylov parameters)
- `alg_svd`: SVD algorithm (default: `LAPACK_DivideAndConquer()`)
- `filename`: JLD2 checkpoint file (default: `"default_dmrg2.jld2"`)
- `save_iters`: which sweeps to save (default: all)
- `verbose`: `0` silent, `1` per-sweep summary, `>1` also per-move lines
  (default: `true`)
- `envs`: environment cache (default: `environments(ψ, H, ψ)`)

Returns `(ψ, envs, E₀)`; see also [`dmrg2`](@ref), [`dmrg1!`](@ref).
"""
function dmrg2!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector{<:Integer};
        alg_eigsolve = _default_alg_eigsolve(true, 16),
        alg_svd = LAPACK_DivideAndConquer(),
        filename::String = "default_dmrg2.jld2",
        save_iters::AbstractVector{<:Integer} = collect(eachindex(truncdims)),
        verbose::Union{Bool, Integer} = true,
        envs = environments(ψ, H, ψ))
    algs = map(truncdims) do D
        DMRG2(; alg_eigsolve, alg_svd, trunc = truncrank(Int(D)))
    end
    return _dmrg_run!("DMRG2", ψ, H, algs, envs; filename, save_iters, verbose)
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

The rationale: two-site updates grow the bond by up to a factor of the physical
dimension per update, so they deliver fast and robust bond growth at small D
where sweeps are cheap; one-site updates are much cheaper per sweep at large D,
where the CBE expansion only has to refresh `delta*D` directions per sweep.

# Arguments / schedule forms
- Two-vector form: `dmrg_mix!(ψ, H, [64, 128, 256], [512, 1024, 1024])` runs
  the first vector with DMRG2 and the second with DMRG1+CBE. Sweep numbering
  and JLD2 checkpoints run continuously across both phases.
- Single-vector form: `dmrg_mix!(ψ, H, truncdims; switch_D = 256)` splits
  `truncdims` at the last entry `≤ switch_D`.

# Keyword arguments
- `alg_eigsolve`: eigensolver shared by both phases (default: adaptive
  [`AdaptiveKrylov`]; pass an explicit `Lanczos(...)` to pin fixed Krylov
  parameters)
- `alg_svd`: SVD algorithm (default: `LAPACK_DivideAndConquer()`)
- `alg_expand`, `delta`: bond-expansion strategy for the one-site phase,
  exactly as in [`dmrg1!`](@ref)
- `filename`: JLD2 checkpoint file (default: `"default_dmrg_mix.jld2"`)
- `save_iters`: which sweeps to save (default: all)
- `verbose`: `0` silent, `1` per-sweep summary, `>1` also per-move lines
- `envs`: environment cache (default: `environments(ψ, H, ψ)`)

Returns `(ψ, envs, E₀)`; see also [`dmrg_mix`](@ref), [`dmrg2!`](@ref),
[`dmrg1!`](@ref).
"""
function dmrg_mix!(
        ψ::AbstractFiniteMPS, H,
        truncdims_2site::AbstractVector{<:Integer},
        truncdims_1site::AbstractVector{<:Integer};
        alg_eigsolve = _default_alg_eigsolve(true, 16),
        alg_svd = LAPACK_DivideAndConquer(),
        alg_expand = OptimalExpand,
        delta::Real = 0.1,
        filename::String = "default_dmrg_mix.jld2",
        save_iters::AbstractVector{<:Integer} = 1:(length(truncdims_2site) + length(truncdims_1site)),
        verbose::Union{Bool, Integer} = true,
        envs = environments(ψ, H, ψ))
    delta >= 0 || throw(ArgumentError("delta must be nonnegative"))
    algs = Union{DMRG, DMRG2}[
        (DMRG2(; alg_eigsolve, alg_svd, trunc = truncrank(Int(D))) for D in truncdims_2site)...,
        (DMRG(; alg_eigsolve, alg_svd, trunc = truncrank(Int(D)),
            alg_expand = _dmrg_expand_alg(alg_expand, Int(D), delta, alg_svd))
            for D in truncdims_1site)...,
    ]
    return _dmrg_run!("DMRG-mix", ψ, H, algs, envs; filename, save_iters, verbose)
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

# Default local eigensolver. Adaptive (`AdaptiveKrylov`): tolerance, Krylov
# dimension and restart count are retuned per local update from the measured
# decay rate, the Galerkin errors and the truncation error — the same
# controller MPSKit's `DMRG` uses by default. `adaptive = false` pins the
# previous fixed one-step `Lanczos` with the given `krylovdim`.
function _default_alg_eigsolve(adaptive::Bool, krylovdim::Integer)
    adaptive && return AdaptiveKrylov(; orth = ModifiedGramSchmidt())
    return Lanczos(;
        krylovdim = Int(krylovdim), maxiter = 1, tol = 1e-8,
        orth = ModifiedGramSchmidt(), eager = true, verbosity = 0
    )
end

# `alg_eigsolve === nothing` → built from `adaptive`/`krylovdim`; an explicitly
# passed solver is used as-is (full manual control).
_resolve_alg_eigsolve(alg_eigsolve, adaptive, krylovdim) =
    something(alg_eigsolve, _default_alg_eigsolve(adaptive, krylovdim))

# Per-sweep expansion algorithm: `OptimalExpand` (the type) triggers the default
# delta-scaled CBE path; an instance is used as-is; a callable `D -> alg` is
# invoked with each sweep's target `D`; `nothing` disables expansion.
function _dmrg_expand_alg(alg_expand, D::Int, delta::Real, alg_svd)
    alg_expand === nothing && return nothing
    alg_expand === OptimalExpand &&
        return OptimalExpand(; trunc = truncrank(ceil(Int, delta * D)), alg_svd)
    alg_expand isa Algorithm && return alg_expand
    alg_expand isa Function && return alg_expand(D)
    return throw(ArgumentError(
        "alg_expand must be `nothing`, the type `OptimalExpand`, an expansion " *
        "algorithm instance (e.g. `OptimalExpand(; trunc = ...)`, " *
        "`SketchedExpand(; trunc = ...)`), or a callable `D -> alg`"
    ))
end

# bond index touched by a local update at `pos` moving in direction `dir`
_move_bond(::DMRG, ::Val{:right}, pos::Int) = pos
_move_bond(::DMRG, ::Val{:left}, pos::Int) = pos - 1
_move_bond(::DMRG2, ::Val{:right}, pos::Int) = pos
_move_bond(::DMRG2, ::Val{:left}, pos::Int) = pos

function _dmrg_log_move(alg, dir::Val{D}, pos::Int, ψ, ϵ_local, ϵ_trunc, wpos::Int, wD::Int) where {D}
    b = _move_bond(alg, dir, pos)
    Db = dim(right_virtualspace(ψ, b))
    arrow = D === :right ? "=>" : "<="
    tag = D === :right ? "SweepL2R" : "SweepR2L"
    @printf(
        "  %s: site %*d %s site %*d | D = %*d | ϵ = %.2e | ϵtr = %.2e | %s\n",
        tag, wpos, b, arrow, wpos, b + 1, wD, Db, ϵ_local, ϵ_trunc,
        Dates.format(now(), "d.u yyyy HH:MM")
    )
    return nothing
end

function _dmrg_max_bond_dim(ψ::AbstractFiniteMPS)
    N = length(ψ)
    N <= 1 && return 1
    return maximum(b -> dim(right_virtualspace(ψ, b)), 1:(N - 1))
end

function _dmrg_run!(
        label::String, ψ::AbstractFiniteMPS, H,
        algs::AbstractVector{<:Union{DMRG, DMRG2}}, envs;
        filename::String, save_iters, verbose
    )
    N = length(ψ)
    niters = length(algs)
    isempty(algs) && throw(ArgumentError("truncdims cannot be empty"))
    Tr = real(scalartype(ψ))
    # DMRG updates sites (n = N), DMRG2 updates bonds (n = N - 1); mixed drivers
    # size the bookkeeping arrays by the larger engine
    n = maximum(alg -> _num_updates(alg, ψ), algs)
    ϵ_locals = ones(Tr, n)      # per-position Galerkin errors (drive adaptive solvers)
    ϵ_truncs = zeros(Tr, n)     # per-position truncation errors of the gauge step
    decay_rates = zeros(n)      # per-position observed eigensolver contraction factors
    ϵ_global = one(Tr)
    allocator = default_allocator(ψ, SerialScheduler())
    timer = TimerOutput()
    wpos = ndigits(N)
    witer = ndigits(niters)
    wD = 4

    E_prev = real(expectation_value(ψ, H, envs))
    start_time, record_start = now(), now()
    Int(verbose) > 0 && println("$label Sweep Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
    Int(verbose) > 0 && flush(stdout)

    for iter in 1:niters
        alg = algs[iter]
        fwd, bwd = _sweep_ranges(alg, ψ)
        # positions updated this sweep; statistics are masked to these, since
        # mixed drivers alternate between site-based (DMRG) and bond-based
        # (DMRG2) indexing
        positions = union(fwd, bwd)

        @timeit timer "L2R sweep" begin
            for pos in fwd
                ψ, ϵ_locals[pos], ϵ_truncs[pos], decay_rates[pos] =
                    local_update!(
                        pos, Val(:right), ψ, H, alg, envs,
                        ϵ_global, ϵ_truncs[pos], decay_rates[pos],
                        iter, timer, allocator
                    )
                ϵ_global = maximum(view(ϵ_locals, positions))
                if Int(verbose) > 1
                    _dmrg_log_move(alg, Val(:right), pos, ψ, ϵ_locals[pos], ϵ_truncs[pos], wpos, wD)
                    flush(stdout)
                end
            end
        end
        @timeit timer "R2L sweep" begin
            for pos in bwd
                ψ, ϵ_locals[pos], ϵ_truncs[pos], decay_rates[pos] =
                    local_update!(
                        pos, Val(:left), ψ, H, alg, envs,
                        ϵ_global, ϵ_truncs[pos], decay_rates[pos],
                        iter, timer, allocator
                    )
                ϵ_global = maximum(view(ϵ_locals, positions))
                if Int(verbose) > 1
                    _dmrg_log_move(alg, Val(:left), pos, ψ, ϵ_locals[pos], ϵ_truncs[pos], wpos, wD)
                    flush(stdout)
                end
            end
        end

        E₀ = @timeit timer "expectation_value" real(expectation_value(ψ, H, envs))
        ΔE = abs(E₀ - E_prev)
        E_prev = E₀
        Dmax = _dmrg_max_bond_dim(ψ)
        wD = max(wD, ndigits(Dmax))
        current_time = now()
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
