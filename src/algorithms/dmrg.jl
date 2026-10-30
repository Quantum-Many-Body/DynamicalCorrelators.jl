# Finite-system DMRG drivers with an explicit per-sweep bond-dimension schedule
# plus tail sweeps: after the schedule is exhausted, sweeping continues at the
# final bond dimension until |ΔE| ≤ etol (at most `max_tailsweeps` extra
# sweeps).
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
a left-to-right plus right-to-left pass), e.g. `truncdims = [64, 64, 128, 512]`
runs two sweeps at D = 64, then one at 128 and one at 512. After the schedule
is exhausted, tail sweeps at the final bond dimension continue until the
per-sweep energy change satisfies `|ΔE| ≤ etol` (at most `max_tailsweeps`
extra sweeps; a warning is issued if the tolerance is not reached).

# Keyword arguments
- `alg_eigsolve`: eigensolver for the one-site effective Hamiltonian
  (default: a fixed one-step `Lanczos(krylovdim = 16, maxiter = 1, tol = 1e-8,
  ModifiedGramSchmidt(), eager = true)`; pass an explicit KrylovKit solver to
  override)
- `alg_svd`: SVD algorithm (default: `SafeDivideAndConquer()`)
- `alg_expand`: bond-expansion strategy — an instance (used as-is for every
  sweep), a callable `D -> alg` (rebuilt from each sweep's target `D`), or
  `nothing` to disable expansion (plain one-site DMRG, which cannot grow the
  bond). The default is the callable
  `D -> CBEExpand(; trunc = truncrank(ceil(Int, 0.1*D)), alg_svd)`, expanding
  by up to 10% of the target `D` before the gauge truncates back; MPSKit's
  `OptimalExpand`/`SketchedExpand`/`RandExpand` are drop-in alternatives
- `etol`: convergence tolerance of the tail sweeps on the per-sweep energy
  change `|ΔE|` (default: `1e-8`)
- `max_tailsweeps`: maximum number of tail sweeps at the final bond dimension
  (default: `20`; `0` disables tail sweeps, restoring the plain fixed schedule)
- `save`: JLD2 checkpointing (default: `true`) — `false` writes nothing,
  `true` writes a rolling checkpoint after every sweep: the latest state
  atomically replaces the previous one in `filename` (keys `ψ`, `E`, `ΔE`,
  `ϵ`, `ϵtrunc`, `D`, `sweep`), so a killed run always leaves the newest
  completed sweep on disk; in addition, after every local update the touched
  site/bond tensors and the sweep bookkeeping are written atomically to an
  update-level checkpoint directory — `basename(filename) * ".updates.d"`
  under the environment manager's storage root when environments are
  disk-backed (the transient scratch side, next to the serialized
  environments), `filename * ".updates.d"` when they live in RAM — so
  rerunning the same call after a kill resumes the interrupted sweep at the
  last completed update instead of restarting it, reusing the surviving
  disk-backed environments (in-RAM environments are simply rebuilt). The
  update-level checkpoint belongs to a specific
  `(filename, truncdims)` schedule; a mismatching call discards it and
  starts fresh. The directory is removed when the run finishes. A vector of
  sweep indices (e.g. `save = [2, 4, 6]`) switches to history mode,
  appending those sweeps as `sweep_<i>_ψ`/… entries (no update-level
  checkpointing in this mode)
- `filename`: JLD2 checkpoint file (default: `"default_dmrg1.jld2"`)
- `verbose`: `0` silent, `1` per-sweep summary, `>1` also per-move lines
  (default: `true`)
- `envs`: environment cache — the default `nothing` builds a
  [`HalfFiniteEnvironments`](@ref); pass MPSKit's `environments(ψ, H, ψ)` to
  run with the reference cache (e.g. for cross-checks). The cache type selects
  the local-update method, the per-sweep energy estimate and the memory
  bookkeeping of the sweep driver
- `disk`: environment disk backing when the driver builds the cache — `false`
  (default) keeps environments in RAM, `true` serializes them to a node-local
  temporary directory (`tempdir()`, the same scheme as FiniteMPS.jl's
  `disk = true`), and a directory string uses that directory instead; the
  update-level checkpoint (see `save`) then lives next to the serialized
  environments. Ignored when `envs` is supplied — the supplied manager's own
  storage decides everything
- `manual_gc`: `HalfFiniteEnvironments` only (default: `false`): a full
  `GC.gc(true)` collection and memory report after every sweep. GC is
  deliberately not triggered per local update — with many states evolving
  concurrently, per-update collections make threads wait at GC safepoints

Returns `(ψ, envs, E₀)`; see also [`dmrg1`](@ref), [`dmrg2!`](@ref).
"""
function dmrg1!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector{<:Integer};
        alg_eigsolve = _default_alg_eigsolve(),
        alg_svd = SafeDivideAndConquer(),
        alg_expand = D -> CBEExpand(; trunc = truncrank(ceil(Int, 0.1 * D)), alg_svd),
        etol::Float64 = 1.0e-8,
        max_tailsweeps::Integer = 20,
        filename::String = "default_dmrg1.jld2",
        save::Union{Bool, AbstractVector{<:Integer}} = true,
        verbose::Union{Bool, Integer} = true,
        disk::Union{Bool, AbstractString} = false,
        manual_gc::Bool = false,
        envs::Union{Nothing,FiniteEnvironments,HalfFiniteEnvironments} = nothing,
        _interrupt_after::Int = 0)
    algs = map(truncdims) do D
        DMRG(;
            alg_eigsolve, alg_svd, trunc = truncrank(Int(D)),
            alg_expand = _dmrg_expand_alg(alg_expand, Int(D))
        )
    end
    envs === nothing && (envs = HalfFiniteEnvironments(ψ, H; disk))
    return _dmrg_run!(
        "DMRG1", ψ, H, algs, envs;
        etol = Float64(etol), max_tailsweeps = Int(max_tailsweeps),
        filename, save, verbose, manual_gc,
        _schedule_id = hash(Int.(truncdims)), _interrupt_after
    )
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
left-to-right plus right-to-left pass); after the schedule, tail sweeps at the
final bond dimension continue until `|ΔE| ≤ etol` (see [`dmrg1!`](@ref)).

# Keyword arguments
Same as [`dmrg1!`](@ref), except that there is no `alg_expand` (the two-site
update grows the bond itself) and `filename` defaults to
`"default_dmrg2.jld2"`.

Returns `(ψ, envs, E₀)`; see also [`dmrg2`](@ref), [`dmrg1!`](@ref).
"""
function dmrg2!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector{<:Integer};
        alg_eigsolve = _default_alg_eigsolve(),
        alg_svd = SafeDivideAndConquer(),
        etol::Float64 = 1.0e-8,
        max_tailsweeps::Integer = 20,
        filename::String = "default_dmrg2.jld2",
        save::Union{Bool, AbstractVector{<:Integer}} = true,
        verbose::Union{Bool, Integer} = true,
        disk::Union{Bool, AbstractString} = false,
        manual_gc::Bool = false,
        envs::Union{Nothing,FiniteEnvironments,HalfFiniteEnvironments} = nothing,
        _interrupt_after::Int = 0)
    algs = map(truncdims) do D
        DMRG2(; alg_eigsolve, alg_svd, trunc = truncrank(Int(D)))
    end
    envs === nothing && (envs = HalfFiniteEnvironments(ψ, H; disk))
    return _dmrg_run!(
        "DMRG2", ψ, H, algs, envs;
        etol = Float64(etol), max_tailsweeps = Int(max_tailsweeps),
        filename, save, verbose, manual_gc,
        _schedule_id = hash(Int.(truncdims)), _interrupt_after
    )
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

Tail sweeps at the final bond dimension (with the one-site engine) run until
`|ΔE| ≤ etol`, exactly as in [`dmrg1!`](@ref).

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
        alg_eigsolve = _default_alg_eigsolve(),
        alg_svd = SafeDivideAndConquer(),
        alg_expand = D -> CBEExpand(; trunc = truncrank(ceil(Int, 0.1 * D)), alg_svd),
        etol::Float64 = 1.0e-8,
        max_tailsweeps::Integer = 20,
        filename::String = "default_dmrg_mix.jld2",
        save::Union{Bool, AbstractVector{<:Integer}} = true,
        verbose::Union{Bool, Integer} = true,
        disk::Union{Bool, AbstractString} = false,
        manual_gc::Bool = false,
        envs::Union{Nothing,FiniteEnvironments,HalfFiniteEnvironments} = nothing,
        _interrupt_after::Int = 0)
    algs = Union{DMRG, DMRG2}[
        (DMRG2(; alg_eigsolve, alg_svd, trunc = truncrank(Int(D))) for D in truncdims_2site)...,
        (DMRG(; alg_eigsolve, alg_svd, trunc = truncrank(Int(D)),
            alg_expand = _dmrg_expand_alg(alg_expand, Int(D)))
            for D in truncdims_1site)...,
    ]
    envs === nothing && (envs = HalfFiniteEnvironments(ψ, H; disk))
    return _dmrg_run!(
        "DMRG-mix", ψ, H, algs, envs;
        etol = Float64(etol), max_tailsweeps = Int(max_tailsweeps),
        filename, save, verbose, manual_gc,
        _schedule_id = hash((Int.(truncdims_2site), Int.(truncdims_1site))),
        _interrupt_after
    )
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

# Default local eigensolver: a fixed one-step `Lanczos` with the given
# `krylovdim` budget for every local update (eager early exit keeps easy
# updates cheap). This is the conventional fixed-budget choice, matching
# e.g. FiniteMPS.jl's LanczosGS(K = 16, tol = 1e-8).
function _default_alg_eigsolve(
        krylovdim::Integer = 16;
        tol::Real = 1e-8, maxiter::Integer = 1, eager::Bool = true
    )
    return Lanczos(;
        krylovdim = Int(krylovdim), maxiter = Int(maxiter), tol = Float64(tol),
        orth = ModifiedGramSchmidt(), eager = eager, verbosity = 0
    )
end

# `alg_eigsolve === nothing` → the fixed default with the given `krylovdim`
_resolve_alg_eigsolve(alg_eigsolve, krylovdim) =
    something(alg_eigsolve, _default_alg_eigsolve(krylovdim))

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

_diag_s(ns::Real) = ns / 1e9   # nanoseconds → seconds, for log formatting

# Per-update log line. With the half engine's `diag` NamedTuple the line
# carries the matvec count K and per-phase wall times (eigsolve split into
# matvec vs Gram–Schmidt/bookkeeping rest, environment rebuild, operator
# construction, CBE expand, gauge), matching the information FiniteMPS.jl logs
# per update (its `K = ...` line and per-step TimerOutput). `λ` is the local
# eigenvalue and `ϵtr` the gauge/SVD truncation error. `diag === nothing` is
# the reference-engine fallback. `exp` is printed only for the one-site/CBE path.
function _dmrg_log_move(alg, dir::Val{D}, pos::Int, ψ, λ, ϵ_trunc, wpos::Int, wD::Int, diag) where {D}
    b = _move_bond(alg, dir, pos)
    # One-site R2L: the gauge just truncated the LEFT leg of AR[pos] (bond
    # pos-1). right_virtualspace(ψ, pos-1) would instead show the CBE-expanded
    # leg of AL[pos-1], which keeps the enlarged space until the next L2R pass
    # re-truncates it — same convention as MPSKit, but confusing in the log.
    Db = alg isa DMRG && D === :left ? dim(left_virtualspace(ψ, pos)) : dim(right_virtualspace(ψ, b))
    arrow = D === :right ? "=>" : "<="
    tag = D === :right ? "L2R" : "R2L"
    timestamp = format(now(), "HH:MM:SS")
    if diag === nothing
        @printf(
            "  %s: %*d%s%*d | D=%*d | ϵtr %.1e | %s\n",
            tag, wpos, b, arrow, wpos, b + 1, wD, Db, ϵ_trunc, timestamp
        )
    else
        env_str = diag.t_env > 1e6 ?
            @sprintf("env %.1fs | ", _diag_s(diag.t_env)) : ""
        expand_str = diag.t_expand > 1e7 ?
            @sprintf(" | exp %.2fs (can %.1f g2 %.1f svd %.1f rec %.1f | gc %.1f al %.1fG)",
                _diag_s(diag.t_expand), _diag_s(diag.t_expand_can),
                _diag_s(diag.t_expand_g2), _diag_s(diag.t_expand_svd),
                _diag_s(diag.t_expand_rec), diag.gc_expand / 1e9,
                diag.alloc_expand / 2^30) : ""
        @printf(
            "  %s: %*d%s%*d | D=%*d | E=%.6f | K=%2d | eig %.1fs (mv %.1fs) | %sham %.1fs%s | gauge %.1fs | ϵtr %.1e | %s\n",
            tag, wpos, b, arrow, wpos, b + 1, wD, Db, real(λ), diag.numops,
            _diag_s(diag.t_eig), _diag_s(diag.t_mv),
            env_str, _diag_s(diag.t_ham), expand_str, _diag_s(diag.t_gauge),
            ϵ_trunc, timestamp
        )
    end
    return nothing
end

# Per-sweep aggregate of the per-update diagnostics plus GC statistics
# (`Base.gc_num` deltas: collection time, allocations, safepoint waits)
struct _SweepDiag
    n_mv::Int        # matvecs this sweep
    t_mv::Float64    # matvec time (ns)
    t_eig::Float64   # eigsolve time (ns)
    t_env::Float64   # zero on the lazy 20823-style env path (ns)
    t_ham::Float64   # effective-Hamiltonian construction (ns; envs already fresh)
    t_expand::Float64
    t_gauge::Float64
    t_expand_can::Float64   # changebond! sub-phases (ns)
    t_expand_g2::Float64
    t_expand_svd::Float64
    t_expand_rec::Float64
    gc_expand::Float64      # expand GC time (ns)
    alloc_expand::Float64   # expand allocations (bytes)
end
_SweepDiag() = _SweepDiag(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
function _accum!(s::_SweepDiag, diag)
    diag === nothing && return s
    return _SweepDiag(
        s.n_mv + diag.numops, s.t_mv + diag.t_mv, s.t_eig + diag.t_eig,
        s.t_env + diag.t_env, s.t_ham + diag.t_ham,
        s.t_expand + diag.t_expand, s.t_gauge + diag.t_gauge,
        s.t_expand_can + diag.t_expand_can, s.t_expand_g2 + diag.t_expand_g2,
        s.t_expand_svd + diag.t_expand_svd, s.t_expand_rec + diag.t_expand_rec,
        s.gc_expand + diag.gc_expand, s.alloc_expand + diag.alloc_expand
    )
end
function _print_sweep_diag(io::IO, s::_SweepDiag, gc0, gc1, sweep_seconds::Float64)
    s.n_mv == 0 && return nothing
    gc_s = (gc1.total_time - gc0.total_time) / 1e9
    safepoint_s = (gc1.total_time_to_safepoint - gc0.total_time_to_safepoint) / 1e9
    alloc_gib = (gc1.total_allocd - gc0.total_allocd) / 2^30
    env_str = s.t_env > 1e6 ?
        @sprintf("env %.1fs | ", _diag_s(s.t_env)) : ""
    expand_str = s.t_expand > 1e7 ?
        @sprintf(" | exp %.2fs (can %.1f g2 %.1f svd %.1f rec %.1f | gc %.1f al %.1fG)",
            _diag_s(s.t_expand), _diag_s(s.t_expand_can), _diag_s(s.t_expand_g2),
            _diag_s(s.t_expand_svd), _diag_s(s.t_expand_rec),
            s.gc_expand / 1e9, s.alloc_expand / 2^30) : ""
    @printf(
        io,
        "  mv %d (%.0f ms avg) | eig %.1fs (mv %.1fs, %.0f%%) | %sham %.1fs%s | gauge %.1fs | GC %.1fs + safepoint %.1fs (%.1f%% of sweep) | alloc %.2f GiB\n",
        s.n_mv, s.t_mv / 1e6 / s.n_mv,
        _diag_s(s.t_eig), _diag_s(s.t_mv), 100 * s.t_mv / max(s.t_eig, eps()),
        env_str, _diag_s(s.t_ham), expand_str, _diag_s(s.t_gauge),
        gc_s, safepoint_s, 100 * (gc_s + safepoint_s) / max(sweep_seconds, eps()),
        alloc_gib
    )
    return flush(io)
end

function _dmrg_run!(
        label::String, ψ::AbstractFiniteMPS, H,
        algs::AbstractVector{<:Union{DMRG, DMRG2}},
        envs::Union{FiniteEnvironments, HalfFiniteEnvironments};
        filename::String, save::Union{Bool, AbstractVector{<:Integer}}, verbose,
        etol::Float64, max_tailsweeps::Int,
        manual_gc::Bool = false,
        _schedule_id::UInt = UInt(0), _interrupt_after::Int = 0
    )
    N = length(ψ)
    isempty(algs) && throw(ArgumentError("truncdims cannot be empty"))
    algs = collect(algs)   # tail sweeps push! into this
    nscheduled = length(algs)
    save_iters = save isa Bool ? Int[] : collect(Int, save)
    all(>=(1), save_iters) || throw(ArgumentError("save indices must be >= 1"))
    wrote = false
    # DMRG updates sites (n = N), DMRG2 updates bonds (n = N - 1); mixed
    # schedules size the bookkeeping arrays by the larger engine
    n = maximum(alg -> alg isa DMRG ? N : N - 1, algs)
    ϵ_truncs = zeros(n)         # per-position truncation errors of the gauge step
    allocator = default_allocator(ψ, SerialScheduler())
    timer = TimerOutput()
    wpos = ndigits(N)      # log column width: site index
    witer = ndigits(nscheduled)  # log column width: sweep counter
    wD = 4                 # log column width: bond dimension (grows with D)

    # update-level checkpoint: resume in place when a compatible mid-sweep
    # checkpoint exists (see algorithms/dmrg_checkpoint.jl); tied to the
    # rolling checkpoint (its base state), hence active iff save === true.
    # The directory follows the environments' storage root (scratch side)
    # when they are disk-backed, `filename` otherwise
    ckpt_active = save === true
    resumed = ckpt_active ?
        _resume_update_checkpoint(filename, ψ, algs, envs, _schedule_id, verbose) :
        nothing
    resume_point = nothing
    iter = 0
    if resumed === nothing
        ckpt_active && rm(_updates_dir(filename, envs); recursive = true, force = true)
        # reference path needs a real energy for the first ΔE; the half
        # engine's per-sweep energy is the last update's λ, so its first ΔE is
        # NaN
        E_prev = envs isa HalfFiniteEnvironments ? NaN : real(expectation_value(ψ, H, envs))
        E₀, ΔE = NaN, NaN
    else
        ψ = resumed.ψ
        ϵ_truncs = resumed.ϵ_truncs
        E_prev = resumed.E_prev
        E₀, ΔE = resumed.E₀, resumed.ΔE
        resume_point = resumed.resume_point
        iter = resumed.done_sweep ? resumed.iter : resumed.iter - 1
    end
    ckpt = ckpt_active ?
        _update_ckpt_context(filename, nscheduled, envs, _schedule_id) : nothing
    λ = 0.0   # assigned inside the sweep loops; pre-declare so it is
              # visible after them (loop bodies are their own scope)
    nupdates = 0
    start_time, record_start = now(), now()
    Int(verbose) > 0 && println(
        "$label Sweep Started (",
        envs isa HalfFiniteEnvironments ? "half engine" : "reference engine",
        "): ", format(start_time, "d.u yyyy HH:MM")
    )
    Int(verbose) > 0 && flush(stdout)
    while iter < length(algs)
        iter += 1
        alg = algs[iter]
        # sweep ranges: DMRG sweeps sites 1:N-1 then N:-1:2, DMRG2 sweeps bonds
        # 1:N-1 then N-2:-1:1 (the last L2R bond was just updated)
        fwd, bwd = alg isa DMRG ? (1:(N - 1), N:-1:2) : (1:(N - 1), (N - 2):-1:1)
        # a resumed sweep re-enters at the recorded position; the fingerprint
        # of its (already past) sweep start is restored rather than recomputed
        if ckpt !== nothing
            ctx_fingerprint = resume_point !== nothing ?
                resumed.base_fingerprint : _state_fingerprint(ψ)
            ckpt.base_fingerprint = ctx_fingerprint
        end
        if resume_point !== nothing
            if resume_point.dir == :right
                fwd = (resume_point.pos + 1):last(fwd)
            else
                fwd = 1:0
                bwd = (resume_point.pos - 1):-1:last(bwd)
            end
            resume_point = nothing
        end
        # positions updated this sweep; statistics are masked to these, since
        # mixed drivers alternate between site-based (DMRG) and bond-based
        # (DMRG2) indexing
        positions = union(fwd, bwd)
        # spectral shift of this sweep's eigensolves: the half engine solves
        # `H - E_prev` (a pure shift leaves the Krylov subspace unchanged)
        shift = envs isa HalfFiniteEnvironments && !isnan(E_prev) ? E_prev : 0.0

        gc0 = Base.gc_num()
        sweepdiag = _SweepDiag()
        @timeit timer "L2R sweep" begin
            for pos in fwd
                nupdates += 1
                _interrupt_after > 0 && nupdates > _interrupt_after &&
                    throw(InterruptException())
                pre = ckpt === nothing ? nothing :
                    (copy(ψ.ALs), copy(ψ.ARs), copy(ψ.ACs), copy(ψ.Cs))
                ψ, λ, ϵ_truncs[pos], diag =
                    _dmrg_update!(
                        pos, Val(:right), ψ, H, alg, envs,
                        ϵ_truncs[pos], iter, timer, allocator, shift
                    )
                sweepdiag = _accum!(sweepdiag, diag)
                envs isa HalfFiniteEnvironments && _free_after_move!(envs, alg, Val(:right), pos)
                if ckpt !== nothing
                    _update_ckpt_state!(
                        ckpt, ψ, envs, pre;
                        engine = nameof(typeof(alg)), iter, dir = :right, pos,
                        done_sweep = false, E₀ = NaN, ΔE = NaN,
                        E_prev, ϵ_truncs, nalgs = length(algs)
                    )
                end
                if Int(verbose) > 1
                    _dmrg_log_move(alg, Val(:right), pos, ψ, λ, ϵ_truncs[pos], wpos, wD, diag)
                    flush(stdout)
                end
            end
        end
        @timeit timer "R2L sweep" begin
            for pos in bwd
                nupdates += 1
                _interrupt_after > 0 && nupdates > _interrupt_after &&
                    throw(InterruptException())
                pre = ckpt === nothing ? nothing :
                    (copy(ψ.ALs), copy(ψ.ARs), copy(ψ.ACs), copy(ψ.Cs))
                ψ, λ, ϵ_truncs[pos], diag =
                    _dmrg_update!(
                        pos, Val(:left), ψ, H, alg, envs,
                        ϵ_truncs[pos], iter, timer, allocator, shift
                    )
                sweepdiag = _accum!(sweepdiag, diag)
                envs isa HalfFiniteEnvironments && _free_after_move!(envs, alg, Val(:left), pos)
                if ckpt !== nothing
                    _update_ckpt_state!(
                        ckpt, ψ, envs, pre;
                        engine = nameof(typeof(alg)), iter, dir = :left, pos,
                        done_sweep = false, E₀ = NaN, ΔE = NaN,
                        E_prev, ϵ_truncs, nalgs = length(algs)
                    )
                end
                if Int(verbose) > 1
                    _dmrg_log_move(alg, Val(:left), pos, ψ, λ, ϵ_truncs[pos], wpos, wD, diag)
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
        witer = max(witer, ndigits(iter))
        current_time = now()
        if manual_gc
            GC.gc(true)
            Int(verbose) > 0 &&
                _print_mem("sweep $iter done", envs isa HalfFiniteEnvironments ? envs : nothing)
        end
        if Int(verbose) > 0
            println(
                "[", lpad(iter, witer), "] ", label, "/",
                nameof(typeof(alg)), " sweep | duration: ",
                canonicalize(current_time - start_time)
            )
            @printf(
                "  E₀ = %.10f | D = %*d | ΔE = %.3e | max ϵtr = %.3e\n",
                E₀, wD, Dmax, ΔE, maximum(view(ϵ_truncs, positions))
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

        # save === true: rolling checkpoint — after EVERY sweep the latest state
        # is written to a temp file that is then atomically renamed onto
        # `filename`, so the file on disk always holds the newest completed
        # sweep and a kill mid-write cannot corrupt it. Fixed keys: "ψ", "E",
        # "ΔE", "ϵtrunc", "D", "sweep" (the sweep index).
        # save = [i, ...]: history mode — those sweeps are appended as
        # sweep_<i>_* entries (one full state each; the file grows accordingly).
        if save === true
            tmp = filename * ".tmp"
            jldopen(tmp, "w") do f
                f["ψ"] = ψ
                f["E"] = E₀
                f["ΔE"] = ΔE
                f["ϵtrunc"] = ϵ_truncs
                f["D"] = Dmax
                f["sweep"] = iter
            end
            mv(tmp, filename; force = true)
            if ckpt !== nothing
                # mark the sweep complete, then drop the per-update tensors —
                # the rolling checkpoint is the authoritative state now
                _update_ckpt_state!(
                    ckpt, ψ, envs, nothing;
                    engine = nameof(typeof(alg)), iter, dir = :right, pos = 0,
                    done_sweep = true, E₀, ΔE, E_prev, ϵ_truncs,
                    nalgs = length(algs)
                )
                _clear_update_tensors(ckpt.dir)
            end
        elseif iter in save_iters
            mode = wrote ? "a" : "w"
            jldopen(filename, mode) do f
                f["sweep_$(iter)_ψ"] = ψ
                f["sweep_$(iter)_E"] = E₀
                f["sweep_$(iter)_ΔE"] = ΔE
                f["sweep_$(iter)_ϵtrunc"] = ϵ_truncs
                f["sweep_$(iter)_D"] = Dmax
            end
            wrote = true
        end
        start_time = current_time
    end

    ckpt !== nothing && rm(ckpt.dir; recursive = true, force = true)
    record_end = now()
    if !(ΔE <= etol) && max_tailsweeps > 0 && iter - nscheduled >= max_tailsweeps
        @warn "$label did not reach |ΔE| ≤ etol = $etol after $max_tailsweeps tail sweep(s) (ΔE = $ΔE)"
    end
    if Int(verbose) > 0
        println(
            "Ended: ", format(record_end, "d.u yyyy HH:MM"),
            " | total duration: ", canonicalize(record_end - record_start),
            " | ", iter, " sweep(s) (", iter - nscheduled, " tail)"
        )
        println(timer)
        _print_krylov_breakdown(timer)
    end
    return ψ, envs, E_prev
end


# Krylov timing breakdown derived from the driver timer. Every application of
# the effective Hamiltonian is timed as the nested "matvec" section inside
# "AC_eigsolve"/"AC2_eigsolve" (finiteengine/localupdate.jl), so
# eigsolve − matvec is the Gram–Schmidt orthogonalization + normalization +
# KrylovKit bookkeeping share — no instrumentation in the vector kernels
function _print_krylov_breakdown(timer::TimerOutput, io::IO = stdout)
    # the eigensolver sections sit one level down, nested under the sweep-direction
    # sections ("L2R sweep"/"R2L sweep") — collect every occurrence
    found = Tuple{String,Any}[]  # TimerOutputs 1.x inner sections are Section, not TimerOutput
    for (sweep, sweepsec) in timer.inner_timers
        for (name, sec) in sweepsec.inner_timers
            name in ("AC_eigsolve", "AC2_eigsolve") && push!(found, (sweep * "/" * name, sec))
        end
    end
    isempty(found) && return nothing
    println(
        io,
        "Krylov breakdown (eigsolve − matvec ≈ Gram–Schmidt orthogonalization,",
        " normalization, KrylovKit bookkeeping):"
    )
    for (name, sec) in found
        t_all = timer_time(sec)
        t_all == 0 && continue
        n_all = timer_ncalls(sec)
        mv = get(sec.inner_timers, "matvec", nothing)
        t_mv = isnothing(mv) ? 0 : timer_time(mv)
        n_mv = isnothing(mv) ? 0 : timer_ncalls(mv)
        t_rest = t_all - t_mv
        f(ns) = ns == 0 ? "0" : string(canonicalize(Millisecond(ns ÷ 1_000_000)))
        pct(x) = round(100 * x / t_all; digits = 1)
        println(
            io, "  ", name, ": ", n_all, " call(s) | total ", f(t_all),
            " | matvec ", f(t_mv), " (", n_mv, " matvec(s), ", pct(t_mv), "%)",
            " | orth+norm+bookkeeping ", f(t_rest), " (", pct(t_rest), "%)"
        )
    end
    return flush(io)
end

# ---------------------------------------------------------------------------
# local-update dispatch
# ---------------------------------------------------------------------------

# The half-engine method (finiteengine/localupdate.jl) returns the local
# eigenvalue λ and a per-update diagnostics NamedTuple, and takes
# `energy_shift`; the reference path is MPSKit's own `local_update!` (λ
# placeholder NaN, unused — its sweep energy comes from `expectation_value` —
# and no diagnostics). MPSKit's method keeps its signature (including the
# `decay_rate` slot, inert for a fixed `Lanczos` solver), so the reference
# branch passes `0.0` and discards the returned value.
function _dmrg_update!(pos, direction::Val, ψ, H, alg,
        envs::HalfFiniteEnvironments, ϵ_trunc,
        iter, timer, allocator, energy_shift::Real)
    return local_update!(
        pos, direction, ψ, H, alg, envs, timer, allocator; energy_shift
    )
end
function _dmrg_update!(pos, direction::Val, ψ, H, alg,
        envs::FiniteEnvironments, ϵ_trunc,
        iter, timer, allocator, ::Real)
    ψ, _, ϵ_trunc, _decay = local_update!(
        pos, direction, ψ, H, alg, envs, 0.0, ϵ_trunc, 0.0,
        iter, timer, allocator
    )
    return ψ, NaN, ϵ_trunc, nothing
end
