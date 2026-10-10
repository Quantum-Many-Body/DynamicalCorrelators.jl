# Time-split (meet-in-the-middle) zero-temperature dynamical correlator.
#
# Instead of evolving one charged ket through the full time grid and projecting
# it onto the t = 0 charged bras (the direct `dcorrelator`), the time interval
# is split in half and both sides of the operator sandwich are evolved to
# τ = t/2:
#
#   greater:  gf[i, j, t] = -im * e^{+i E0 t} * <a_i(t/2) | b_j(t/2)>
#   lesser:   gf[i, j, t] = -im * e^{-i E0 t} * conj(<e_i(t/2) | d_j(t/2)>)
#
#   |a_i(τ)> = e^{+iHτ} c†_i |0>,   |b_j(τ)> = e^{-iHτ} c†_j |0>   (block = :greater)
#   |e_i(τ)> = e^{+iHτ} c_i  |0>,   |d_j(τ)> = e^{-iHτ} c_j  |0>   (block = :lesser)
#
# Every state is only evolved up to τ = tmax/2, halving the peak entanglement
# and the accumulated truncation error compared to the direct method. The price
# is an all-pairs overlap matrix per time slice and 2L resident states per
# block instead of L. Backward evolution (e^{+iHτ}) is simply a TDVP step with
# negative dt; TDVP makes no assumption on the sign of dt (its center-bond
# step already uses -dt/2 internally).

"""
    dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap;
                times, block, kwargs...)

Zero-temperature dynamical correlator with the time-split (meet-in-the-middle)
scheme. Both sides of the operator sandwich are evolved to half the requested
time:

    greater (block = :greater, op = creation operator):
        gf[i, j, r] = -im * e^{+i E0 tᵣ} * <a_i(tᵣ/2) | b_j(tᵣ/2)>
    lesser (block = :lesser, op = annihilation operator):
        gf[i, j, r] = -im * e^{-i E0 tᵣ} * conj(<e_i(tᵣ/2) | d_j(tᵣ/2)>)

with `|a_i(τ)> = e^{+iHτ} c†_i|0>`, `|b_j(τ)> = e^{-iHτ} c†_j|0>` (greater) and
`|e_i(τ)> = e^{+iHτ} c_i|0>`, `|d_j(τ)> = e^{-iHτ} c_j|0>` (lesser). The
phase/conjugation conventions match the direct method exactly, so the retarded
combination is assembled as before: `gfxt = gf_greater .+ gf_lesser`.

This three-positional-argument signature unambiguously selects the time-split
method; the direct methods take a fourth positional `id::Integer` or
`indices::AbstractArray`.

# Differences from the direct method
- The full `length(H) × length(H)` matrix of site pairs is computed at every
  slice; there is no `indices` argument — any missing row or column leaves the
  correlator incomplete, so the method always computes the whole matrix.
- Each call computes one block. `block = :greater` takes the creation operator
  (charged states in the N+1 sector), `block = :lesser` the annihilation
  operator (N-1 sector). The two blocks are fully independent and can run as
  separate jobs.
- `2 * length(H)` fat states are kept in memory (both time directions of one
  block), versus `length(H)` kets plus cheap t = 0 bras in the direct method.
  The TDVP environments dominate the footprint; `disk = true` (or a directory)
  pages them to disk as usual, one random subdirectory per state.

# Keywords
- `block::Symbol` (required): `:greater` or `:lesser`.
- `times = 0:0.05:5.0`: the correlator time grid. Internally the states evolve
  on `times ./ 2`, so the TDVP step is half the grid spacing.
- `record_indices`: continuous window into `times`; evolution stops at the
  last recorded slice.
- `parallel_charged`, `parallel_tdvp2`, `parallel_tdvp1` (each defaults to
  `Threads.nthreads()`):
  per-phase caps on concurrent tasks. The chargedMPS compression and the
  first `n` two-site TDVP steps carry the large per-task workspaces (set by
  `approxalg` / `tdvp2`), while the hundreds of one-site steps that
  dominate the wall time are memory-light — so launch Julia with a large
  thread pool and throttle only the memory-hungry phases, e.g.
  `parallel_charged ≈ 0.7 * RAM / per-compression-workspace`. During
  evolution the bra and ket families step forward as one merged pool of
  `2 * length(H)` tasks, so a single slow state idles at most one slot
  instead of stalling a whole family.
- `chargedmps_path`: directory of cached charged states (default `nothing`
  = cache off). Files are named `<tag>_i=NNN.jld2` with the zero-padded
  cache index NNN = site for `block = :greater` and `L + site` for
  `block = :lesser`, so one directory serves both blocks of the same
  model; `tag` is derived from `approxalg` (truncation, tol, maxiter). A
  present, readable file is used as-is: the cache is never validated
  against the model, `gs` or `op` — picking a matching directory is the
  caller's responsibility. Missing or unreadable files are recomputed and
  written back atomically (temp file + rename), so an interrupted run
  resumes the charged phase from the completed sites only.
- `hamiltonian_threads` (default `nothing` = automatic): cap on the fan-out
  of the channel-split threaded Hamiltonian inside each matvec. With
  `nothing`, the cap is derived per phase as
  `max(1, Threads.nthreads() ÷ outer concurrency)`, so the chargedMPS phase
  (few fat tasks) gets inner threading while the evolution phases (2L tasks
  in flight) run their matvecs serially — outer × inner never exceeds the
  thread pool. An integer overrides all phases with a fixed cap. The global
  cap is restored to its previous value when the call ends.
- `mode` (default `:threaded`): `:threaded` runs everything in this process
  as described above. `:distributed` instead treats the `2L` states as
  independent tasks over the existing `Distributed.workers()` — one state
  per worker process — communicating only through checkpoint files under
  `workdir` (required; fast shared storage). Workers are never launched by
  the package: add them in the driver script with the project activated and
  a single thread (`exeflags = "--project=<env> -t1"`), using whatever
  manager matches the cluster — `LocalAffinityManager(np; mode = COMPACT)`
  on one node, `SlurmManager(np)` / `LSFManager(np)` from ClusterManagers.jl
  for multi-node; the driver discovers the worker → host mapping itself and
  splits the states evenly across hosts, so any of them works unchanged.
  Two strict phases run: first the charged initial states (the memory peak
  of the whole run, throttled by `charged_per_node`), then all evolutions
  with one state per worker. Process isolation removes the shared-heap
  stop-the-world GC coupling that dominates the threaded driver at high
  thread counts, and every recorded slice is checkpointed, so a killed job
  (walltime, node failure) resumes from the last complete slice. The
  threaded-only knobs (`parallel_charged`/`parallel_tdvp2`/
  `parallel_tdvp1`, `hamiltonian_threads`) are ignored in this mode.
- `charged_per_node` (default `nothing`, distributed mode only): how many
  chargedMPS computations may run concurrently per node during phase 1.
  Size it as node memory ÷ per-instance peak (e.g. ~8–12 for D = 5000 on a
  384 GB node). `nothing` lifts the cap — fine when the chargedMPS cache
  hits (loads take seconds), dangerous on a cold cache: dozens of
  simultaneous compressions will exceed any node's memory. Ignored in
  `:threaded` mode.
- `backlog` (default 2, distributed mode): how many recorded slices workers
  may run ahead of the master's overlap pass; at most `backlog + 1` slices
  (each `2L` states) occupy `workdir` at any time.
- `overlap_block` (default 8, distributed mode): tile size of the master's
  thread-parallel overlap pass — at most `2 * overlap_block` states are
  resident on the master.
- `keep_final_states` (default `true`, distributed mode): keep the last
  slice's checkpoints (the τ = tmax/2 states) for reuse; `false` deletes
  them.
- `save`, `gf_path`, `disk`, `n`, `approxalg`, `tdvp1`, `tdvp2`: same meaning
  as in the direct `dcorrelator`.

The charged-state setup, state evolutions, and all-pairs overlaps are
parallelized with `Threads.@threads :dynamic` (dynamic scheduling:
per-compression, per-state evolution, and per-pair overlap costs are all
far from uniform). Start Julia with several threads and keep
`BLAS.set_num_threads` small to avoid oversubscription.

With `verbose = true`, every recorded slice logs a small diagnostics block
after the usual progress line, organized as an indented wall-clock tree:
the serial phases (evolve / overlaps / other) add up to the slice duration;
under `evolve`, the per-state breakdown of the TDVP sections (Krylov
integration, Hamiltonian construction, gauge/expand, disk-backed
environment reads, and the untimed remainder) followed by the
compute / GC / safepoint-wait split of the phase wall. These tables are the
first place to look when a slice is slower than expected.

Returns `gf :: Array{ComplexF64,3}` of size
`(length(H), length(H), length(record_indices))`. With `save = true`, each
recorded slice is appended to
`gf_split_start=<t0>_end=<t1>_block=<block>.jld2` under `pro_<r>`, together
with the grid metadata.

Note: the low-lying-state (Lehmann) removal of Sec. 7.3 is not wired into this
method yet; it will arrive as a separate `dcorrelator` method.
"""
function dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap;
                     block::Symbol,
                     times::AbstractVector=0:0.05:5.0,
                     verbose::Bool=true,
                     save::Bool=true,
                     gf_path::String="./",
                     chargedmps_path::Union{Nothing,AbstractString}=nothing,
                     disk::Union{Bool,AbstractString}=false,
                     record_indices::AbstractArray=1:length(times),
                     parallel_charged::Integer=Threads.nthreads(),
                     parallel_tdvp2::Integer=Threads.nthreads(),
                     parallel_tdvp1::Integer=Threads.nthreads(),
                     hamiltonian_threads::Union{Nothing,Integer}=nothing,
                     mode::Symbol=:threaded,
                     workdir::Union{Nothing,AbstractString}=nothing,
                     backlog::Integer=2,
                     overlap_block::Integer=8,
                     keep_final_states::Bool=true,
                     charged_per_node::Union{Nothing,Integer}=nothing,
                     n::Integer=3,
                     approxalg=myDMRG2(),
                     tdvp1=myTDVP1(),
                     tdvp2=myTDVP2(; trunc=truncerror(; rtol=1e-3)))
    block in (:greater, :lesser) ||
        throw(ArgumentError("block must be :greater or :lesser, got :$block"))
    mode in (:threaded, :distributed) ||
        throw(ArgumentError("mode must be :threaded or :distributed, got :$mode"))
    mode === :distributed && return _ts_dist_dcorrelator(
        gs, H, op; block, times, verbose, save, gf_path, chargedmps_path,
        disk, record_indices, n, approxalg, tdvp1, tdvp2, workdir, backlog,
        overlap_block, keep_final_states, charged_per_node)
    parallel_charged >= 1 && parallel_tdvp2 >= 1 && parallel_tdvp1 >= 1 ||
        throw(ArgumentError("parallel_charged, parallel_tdvp2 and parallel_tdvp1 must all be >= 1"))
    hamiltonian_threads === nothing || hamiltonian_threads >= 1 ||
        throw(ArgumentError("hamiltonian_threads must be >= 1 (or nothing for automatic)"))
    save && !isdir(gf_path) && mkpath(gf_path)
    L = length(H)
    record_indices, record_first, record_last = _dcorrelator_record_window(times, record_indices)
    nrec = length(record_indices)
    gsenergy = expectation_value(gs, H)

    timer = TimerOutput()
    gf = zeros(ComplexF64, L, L, nrec)
    filename = joinpath(gf_path,
                        "gf_split_start=$(times[record_first])_end=$(times[record_last])" *
                        "_block=$(block).jld2")

    # one spawned task per site; the compression itself is throttled by the
    # semaphore (the workspace sized by approxalg sets this phase's memory
    # high-water mark), while cache loads bypass it. The results are
    # collected with comprehensions, which converge to the concrete element
    # type at runtime — type-stable storage without computing any
    # compression twice
    charged_sem = Base.Semaphore(min(parallel_charged, L))
    cache_tag = _charged_alg_tag(approxalg)
    cache_offset = block === :greater ? 0 : L
    chargedmps_path === nothing || mkpath(chargedmps_path)
    # phase-aware cap for the channel-split Hamiltonian (see the
    # `hamiltonian_threads` keyword): restore the global setting on exit
    hamcap_prev = threaded_hamiltonian_threads()
    try
        set_threaded_hamiltonian!(_ts_hamiltonian_cap(min(parallel_charged, L),
                                                      hamiltonian_threads))
        verbose && println("[threaded Hamiltonian] matvec fan-out cap: $(threaded_hamiltonian_threads())",
                           " (chargedMPS phase, $(min(parallel_charged, L)) concurrent compressions)")
        @timeit timer "setup / chargedMPS" begin
            ctasks = map(1:L) do i
                Threads.@spawn begin
                    t0 = time_ns()
                    ψ = chargedmps_path === nothing ? nothing :
                        _charged_cache_load(_charged_cache_file(chargedmps_path, cache_tag,
                                                                cache_offset + i))
                    from_cache = ψ !== nothing
                    if !from_cache
                        Base.acquire(charged_sem)
                        try
                            ψ = complex(chargedMPS(op, gs, i, approxalg; disk))
                        finally
                            Base.release(charged_sem)
                        end
                        chargedmps_path === nothing ||
                            _charged_cache_save(_charged_cache_file(chargedmps_path, cache_tag,
                                                                    cache_offset + i), ψ, i, approxalg)
                    end
                    if verbose
                        println("[chargedMPS] site $i/$L | ", from_cache ? "cache" : "computed",
                                " | maxD = ", _ts_max_bonddim((ψ,)), " | ",
                                _short_duration(Dates.Nanosecond(time_ns() - t0)))
                        flush(stdout)
                    end
                    (ψ, copy(ψ), from_cache)
                end
            end
            # fetch just reads the stored result (nothing recomputes); the
            # comprehension arrays converge to the concrete element type, and
            # the task vector is released right after
            kets = [fetch(t)[1] for t in ctasks]   # e^{-iHτ} family (positive dt)
            bras = [fetch(t)[2] for t in ctasks]   # e^{+iHτ} family (negative dt)
            n_loaded = count(t -> fetch(t)[3], ctasks)
            ctasks = nothing
        end
        if verbose && chargedmps_path !== nothing
            println("[chargedMPS cache] loaded $n_loaded/$L, recomputed $(L - n_loaded)",
                    " | dir: ", chargedmps_path)
            flush(stdout)
        end
        @timeit timer "setup / environments" begin
            etasks = map(1:L) do i
                Threads.@spawn (HalfFiniteEnvironments(bras[i], H; disk),
                                HalfFiniteEnvironments(kets[i], H; disk))
            end
            envs_bra = [fetch(t)[1] for t in etasks]
            envs_ket = [fetch(t)[2] for t in etasks]
            etasks = nothing
        end
        timers_bra = [TimerOutput() for _ in bras]
        timers_ket = [TimerOutput() for _ in kets]
        # same per-phase caps for the evolution: the first n two-site steps grow
        # the bonds up to the tdvp2 cap and are workspace-heavy; the one-site
        # steps that dominate the wall time are memory-light. Both families
        # advance as one merged 2L-task pool per slice
        tdvp2_sem = Base.Semaphore(min(parallel_tdvp2, 2L))
        tdvp1_sem = Base.Semaphore(min(parallel_tdvp1, 2L))
        step_durations = Vector{Float64}(undef, 2L)
    
        dest = zeros(ComplexF64, L, L)
        if save
            jldopen(filename, "w") do f
                f["times"] = times
                f["record_indices"] = record_indices
                f["block"] = String(block)
                f["gsenergy"] = gsenergy
            end
        end
    
        wall_start = now()
        verbose && _progress_start(1, length(times),
                                   "time-split correlator: $block block, $L × $L")
        verbose && flush(stdout)
    
        # τ = 0 slice
        slice_start = now()
        slice_start_ns = time_ns()
        if record_first == 1
            @timeit timer "slice 1 / overlaps" _ts_overlap!(dest, bras, kets)
            _ts_assemble!(gf, dest, 1, block, gsenergy, times[1])
            save && jldopen(filename, "a") do f
                f["pro_1"] = gf[:, :, 1]
            end
            verbose && _progress_step(1, length(times), "t = $(times[1])", now() - slice_start)
            verbose && flush(stdout)
            slice_start = now()
            slice_start_ns = time_ns()
        end
    
        for k in 2:record_last
            alg = k > n ? tdvp1 : tdvp2
            sem = k > n ? tdvp1_sem : tdvp2_sem
            set_threaded_hamiltonian!(_ts_hamiltonian_cap(
                min(k > n ? parallel_tdvp1 : parallel_tdvp2, 2L), hamiltonian_threads))
            dτ = (times[k] - times[k - 1]) / 2
            # per-phase GC snapshots: the report below decomposes each serial
            # phase's wall into compute / GC / safepoint wait
            gc_evolve_before = _gc_stamp()
            evolve_start_ns = time_ns()
            @timeit timer "evolve" _ts_evolve!(bras, envs_bra, timers_bra,
                                               kets, envs_ket, timers_ket,
                                               H, dτ, alg, sem, step_durations)
            evolve_ns = time_ns() - evolve_start_ns
            gc_evolve_after = _gc_stamp()
            overlap_ns = 0
            gc_overlap_after = gc_evolve_after
            if k >= record_first
                r = k - record_first + 1
                overlap_start_ns = time_ns()
                @timeit timer "overlaps" _ts_overlap!(dest, bras, kets)
                overlap_ns = time_ns() - overlap_start_ns
                gc_overlap_after = _gc_stamp()
                _ts_assemble!(gf, dest, r, block, gsenergy, times[k])
                save && jldopen(filename, "a") do f
                    f["pro_$(r)"] = gf[:, :, r]
                end
            end
            if verbose
                _progress_step(k, length(times),
                               "t = $(times[k]) | maxD bra = $(_ts_max_bonddim(bras)), ket = $(_ts_max_bonddim(kets))",
                               now() - slice_start)
                _ts_step_report(stdout, step_durations, L, timers_bra, timers_ket,
                                gc_evolve_before, gc_evolve_after, gc_overlap_after,
                                time_ns() - slice_start_ns,
                                evolve_ns, overlap_ns)
                flush(stdout)
            end
            # per-slice reporting only: drop the accumulated TDVP sections so the
            # next table describes the next slice, not the whole run so far
            foreach(reset_timer!, timers_bra)
            foreach(reset_timer!, timers_ket)
            slice_start = now()
            slice_start_ns = time_ns()
        end
        # NOTE: names first assigned inside a `try` are NOT visible after it
        # (Julia scopes try-locals to the block), so the cleanup and the final
        # report must live inside the try, not after the finally
        bras = kets = nothing
        envs_bra = envs_ket = nothing
        GC.gc()
        verbose && _progress_end(wall_start, timer)
    finally
        set_threaded_hamiltonian!(hamcap_prev)
    end

    return gf
end

# evolve both families one TDVP step: one merged 2L-task pool (dt < 0 for
# the e^{+iHτ} bras, dt > 0 for the e^{-iHτ} kets), thread-parallel with at
# most `sem` in flight, each state with its own TimerOutput. `durations[s]`
# receives the wall time of task s (s <= L: bra s, else ket s - L),
# measured after the semaphore is acquired, so it excludes queueing
function _ts_evolve!(bras, envs_bra, timers_bra, kets, envs_ket, timers_ket,
                     H, dτ, alg, sem, durations)
    L = length(bras)
    Threads.@threads :dynamic for s in 1:(2L)
        Base.acquire(sem)
        try
            t0 = time_ns()
            if s <= L
                bras[s], envs_bra[s] = fast_timestep!(bras[s], H, 0, -dτ, alg,
                                                      envs_bra[s];
                                                      timeroutput=timers_bra[s])
            else
                i = s - L
                kets[i], envs_ket[i] = fast_timestep!(kets[i], H, 0, +dτ, alg,
                                                      envs_ket[i];
                                                      timeroutput=timers_ket[i])
            end
            durations[s] = Float64(time_ns() - t0)
        finally
            Base.release(sem)
        end
    end
    return nothing
end

# ---- per-slice diagnostics ----

# cumulative GC/safepoint counters plus current RSS, for per-slice diffs
function _gc_stamp()
    g = Base.gc_num()
    return (time=g.total_time, safepoint=g.total_time_to_safepoint,
            full=g.full_sweep, allocd=g.total_allocd, rss=_rss_bytes())
end

_ts_ns_str(ns) = _short_duration(Dates.Nanosecond(round(Int, ns)))

# aggregate the timed sections of every per-state TDVP timer by name,
# summed over all states; returns name => (ncalls, ns). The two sweep
# wrappers are skipped — their children already account for the time
function _ts_section_stats(timer_sets...)
    stats = Dict{String,Tuple{Int,Int}}()
    for tos in timer_sets, to in tos
        _ts_accum_sections!(stats, to)
    end
    return stats
end

function _ts_accum_sections!(stats, section)
    for (name, child) in section.inner_timers
        if name != "L2R sweep" && name != "R2L sweep"
            calls, ns = get(stats, name, (0, 0))
            stats[name] = (calls + TimerOutputs.ncalls(child),
                           ns + TimerOutputs.time(child))
        end
        _ts_accum_sections!(stats, child)
    end
    return stats
end

# one-line decomposition of a serial phase into compute / GC / safepoint
# wait — the three PARTITION the phase wall (never overlap): GC is
# stop-the-world, the safepoint wait is the stall between "GC triggered" and
# "all threads parked", and compute ≈ phase − GC − safepoint is the
# remainder (a slight underestimate: stragglers finish real work during a
# safepoint wait, and barrier idle lands in compute)
function _ts_phase_gc_line(io, gc_before, gc_after, phase_ns)
    gc_ns = gc_after.time - gc_before.time
    sp_ns = gc_after.safepoint - gc_before.safepoint
    compute_ns = max(phase_ns - gc_ns - sp_ns, 0.0)
    println(io, "        compute ", _ts_ns_str(compute_ns),
            " (", @sprintf("%.1f%%", 100 * compute_ns / phase_ns), ")",
            " | GC ", _ts_ns_str(gc_ns),
            " (", @sprintf("%.1f%%", 100 * gc_ns / phase_ns), ")",
            " | safepoint ", _ts_ns_str(sp_ns),
            " (", @sprintf("%.1f%%", 100 * sp_ns / phase_ns), ")",
            " | full sweeps ", gc_after.full - gc_before.full,
            " | allocated ", _format_gib(gc_after.allocd - gc_before.allocd))
    return nothing
end

# per-slice report printed after the progress line (indented tree; a child's
# % is relative to its parent):
#   evolve / overlaps / other — serial phases, ADD UP to the slice duration;
#   under evolve — the per-state TDVP sections (averages over the 2L
#   concurrently-evolving states; timers include GC/safepoint stops), then
#   the compute/GC/safepoint split of the phase wall;
#   overlaps and other carry the same split as a single line
function _ts_step_report(io, durations, L, timers_bra, timers_ket,
                         gc_evolve_before, gc_evolve_after, gc_overlap_after,
                         step_wall_ns, evolve_ns, overlap_ns)
    println(io, "      evolve   ", _ts_ns_str(evolve_ns),
            " (", @sprintf("%.1f%%", 100 * evolve_ns / step_wall_ns), ")")
    total = sum(durations)   # ratio base only (the Σ cancels against per-state)
    stats = _ts_section_stats(timers_bra, timers_ket)
    attributed = sum(last, values(stats); init=0)
    rows = sort!(collect(stats); by=last ∘ last, rev=true)
    if !isempty(rows)
        namew = max(maximum(length(first(r)) for r in rows), length("unattributed"))
        for (name, (calls, ns)) in rows
            println(io, "        ", rpad(name, namew), "  ",
                    lpad(@sprintf("%.1f%%", 100 * ns / total), 5), "  (",
                    _ts_ns_str(ns / (2L)), "/state | ",
                    _ts_ns_str(ns / max(calls, 1)), "/call, ", calls, " calls)")
        end
        unatt = max(total - attributed, 0.0)
        println(io, "        ", rpad("unattributed", namew), "  ",
                lpad(@sprintf("%.1f%%", 100 * unatt / total), 5), "  (",
                _ts_ns_str(unatt / (2L)), "/state, untimed overhead)")
    end
    _ts_phase_gc_line(io, gc_evolve_before, gc_evolve_after, evolve_ns)
    if overlap_ns != 0
        println(io, "      overlaps ", _ts_ns_str(overlap_ns),
                " (", @sprintf("%.1f%%", 100 * overlap_ns / step_wall_ns), ")")
        _ts_phase_gc_line(io, gc_evolve_after, gc_overlap_after, overlap_ns)
    end
    other_ns = max(step_wall_ns - evolve_ns - overlap_ns, 0.0)
    println(io, "      other    ", _ts_ns_str(other_ns),
            " (", @sprintf("%.1f%%", 100 * other_ns / step_wall_ns),
            ") — assemble/save | RSS: ", _format_gib(gc_overlap_after.rss))
    return nothing
end

# all-pairs Gram matrix dest[ii, jj] = <bras[ii] | kets[jj]>, thread-parallel
function _ts_overlap!(dest, bras, kets)
    Threads.@threads :dynamic for p in CartesianIndices(dest)
        dest[p] = dot(bras[p[1]], kets[p[2]])
    end
    return dest
end

# assemble one recorded slice with the same phase/conjugation convention as
# the direct method
function _ts_assemble!(gf, dest, r, block::Symbol, gsenergy, t)
    phase = block === :greater ? exp(im * gsenergy * t) : exp(-im * gsenergy * t)
    sub = view(gf, :, :, r)
    if block === :greater
        @. sub = -im * phase * dest
    else
        @. sub = -im * phase * conj(dest)
    end
    return gf
end

# per-phase fan-out cap for the channel-split Hamiltonian: automatic mode
# divides the thread pool by the phase's outer concurrency so that
# outer × inner never exceeds the pool; an explicit value overrides
_ts_hamiltonian_cap(outer::Integer, override::Nothing) =
    max(1, Threads.nthreads() ÷ outer)
_ts_hamiltonian_cap(::Integer, override::Integer) = Int(override)

# largest virtual bond dimension over a set of states (for progress logs)
function _ts_max_bonddim(states)
    m = 0
    for ψ in states
        for i in 1:length(ψ)
            m = max(m, dim(left_virtualspace(ψ, i)))
        end
    end
    return m
end

# ---- human-managed charged-state cache ----
# The directory encodes the model / gs / op / approxalg by convention and is
# never validated against any of them: a present, readable file is used
# as-is. Files are named `<tag>_i=NNN.jld2` with the zero-padded cache index
# NNN = site (block = :greater) or L + site (block = :lesser), so a single
# directory serves both blocks of the same model.

# compact label for a truncation strategy (purely a filename tag)
_trunc_label(t::TruncationByOrder) = "r$(t.howmany)"
function _trunc_label(t::TruncationByError)
    parts = String[]
    iszero(t.atol) || push!(parts, "a$(_num_label(t.atol))")
    iszero(t.rtol) || push!(parts, "e$(_num_label(t.rtol))")
    t.p == 2 || push!(parts, "p$(_num_label(t.p))")
    return isempty(parts) ? "e0" : join(parts, "-")
end
_trunc_label(t::TruncationIntersection) = join(map(_trunc_label, t.components), "-")
_trunc_label(t) = _sanitize_label(repr(t))

_num_label(x) = replace(string(x), r"[^0-9A-Za-z.\-]" => "")
_sanitize_label(s::AbstractString) = replace(s, r"[^0-9A-Za-z.\-]+" => "-")

# tag from the accuracy knobs of the compression algorithm (`DMRG2` keeps
# its truncation inside `alg_gauge`); purely a filename label, never
# validated. Falls back to a sanitized `repr` for exotic algorithms
function _charged_alg_tag(alg)
    parts = String[]
    if hasproperty(alg, :alg_gauge) && hasproperty(alg.alg_gauge, :trunc)
        push!(parts, _trunc_label(alg.alg_gauge.trunc))
    end
    hasproperty(alg, :tol) && push!(parts, "tol$(_num_label(alg.tol))")
    hasproperty(alg, :maxiter) && push!(parts, "it$(alg.maxiter)")
    return isempty(parts) ? _sanitize_label(repr(alg)) : join(parts, "_")
end

_charged_cache_file(dir::AbstractString, tag::AbstractString, idx::Integer) =
    joinpath(dir, "$(tag)_i=$(lpad(idx, 3, '0')).jld2")

# load a cached charged state; `nothing` on a missing or unreadable file
# (e.g. torn by a killed job) — the caller then recomputes it
function _charged_cache_load(file::AbstractString)
    isfile(file) || return nothing
    try
        return jldopen(file, "r") do f
            read(f, "state")
        end
    catch
        return nothing
    end
end

# atomic write (temp file + rename): a killed job never leaves a torn file
function _charged_cache_save(file::AbstractString, ψ, site::Integer, alg)
    tmp = "$(file).tmp.$(getpid())-$(Threads.threadid())"
    jldopen(tmp, "w") do f
        f["state"] = ψ
        # provenance metadata, informational only — never read back
        f["site"] = site
        f["created"] = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        f["approxalg"] = repr(alg)
    end
    return mv(tmp, file; force=true)
end
