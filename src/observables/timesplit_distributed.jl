# Distributed (process-per-state) driver for the time-split correlator
# (`dcorrelator(...; mode = :distributed)`).
#
# The threaded driver evolves all 2L states of a block concurrently in one
# process; at tens of threads the shared-heap, stop-the-world GC ends up
# dominating the wall time. This driver instead treats the 2L states as
# independent tasks over the existing `Distributed.workers()` — one state
# per worker process — so every GC is small and process-local and never
# stalls a sibling.
#
# Scheduling is deterministic and scheduler-agnostic. The master discovers
# the worker -> host mapping itself (`gethostname` on every worker; nothing
# here depends on how the workers were launched — SlurmManager, LSFManager,
# LocalAffinityManager, ... all look the same), splits the 2L states across
# hosts proportionally to their worker counts (round-robin within the
# quotas), and then drives two strict phases:
#   1. charged initial states — the memory peak of the whole run, so at
#      most `charged_per_node` computations run concurrently per host
#      (cache hits skip the gate); every result is checkpointed as slice 0;
#   2. evolutions — one state per worker, checkpointing after each slice.
# A worker whose process dies has its task re-queued on a spare worker of
# the same host and resumes from its last checkpoint; a genuine task error
# aborts the run (checkpoints make resubmission cheap).
#
# All worker/master communication is through the shared filesystem: after
# each recorded slice, every worker writes its state to
# `workdir/slice_NNNN/state_SSS.jld2` (atomic temp-file + rename). The
# master polls for a complete slice, computes the L × L overlap matrix
# (tiled, thread-parallel), stores the slice atomically under
# `workdir/meta/pro_NNNN.jld2`, appends it to the same gf file the threaded
# driver writes, and deletes the previous slice. Workers may run at most
# `backlog` slices ahead of the overlap progress (marker
# `meta/overlap_done`), so at most `backlog + 1` slices — a few hundred GB
# at production sizes — occupy the disk at any time.
#
# Crash/resume: killing the job at any point (walltime, node failure, a torn
# gf file) loses at most the slice currently being evolved; rerunning the
# same script resumes every state from its last checkpoint and the overlaps
# from the first missing slice, rebuilding the gf file from the per-slice
# meta files if needed. `meta/config.jld2` guards against resuming with a
# different time grid/block; the physics inputs (model, gs, op) stay
# human-managed by directory convention, exactly like the chargedMPS cache.
#
# The package never launches workers. Driver scripts do, e.g. for SLURM:
#   using Distributed, ClusterManagers
#   addprocs(SlurmManager(72); exeflags = "--project=<env> -t1")
# with two whole nodes reserved (`#SBATCH -N 2 --exclusive`). The master
# only orchestrates and computes overlaps; its thread count (`julia -t M`)
# drives the overlap tiles, a modest M suffices.

# ---- paths, checkpoints, markers ----

_ts_dist_slice_dir(workdir, r) = joinpath(workdir, "slice_$(lpad(r, 4, '0'))")
_ts_dist_state_file(workdir, r, s) =
    joinpath(_ts_dist_slice_dir(workdir, r), "state_$(lpad(s, 3, '0')).jld2")
_ts_dist_meta_dir(workdir) = joinpath(workdir, "meta")
_ts_dist_gf_slice_file(meta_dir, r) = joinpath(meta_dir, "pro_$(lpad(r, 4, '0')).jld2")

# highest record slice with a checkpoint of state s (0 = none); per-state
# writes are sequential and deletion removes whole old slices, so the
# existing checkpoints of a state always form a suffix
function _ts_dist_state_progress(workdir, nrec, s)
    for r in nrec:-1:1
        isfile(_ts_dist_state_file(workdir, r, s)) && return r
    end
    return 0
end

# atomic write of a state checkpoint (temp file + rename — a killed writer
# never leaves a readable torn file)
function _ts_dist_save_state(workdir, r, s, ψ)
    file = _ts_dist_state_file(workdir, r, s)
    tmp = "$(file).tmp.$(getpid())"
    jldopen(tmp, "w") do f
        f["state"] = ψ
        f["slice"] = r
        f["task"] = s
    end
    mv(tmp, file; force=true)
    return file
end

_ts_dist_load_state(workdir, r, s) =
    jldopen(_ts_dist_state_file(workdir, r, s), "r") do f
        read(f, "state")
    end

# master's overlap progress: the plain-text integer in meta/overlap_done
# (atomically replaced) — workers consult it for the disk-usage throttle
function _ts_dist_overlap_done(meta_dir)
    file = joinpath(meta_dir, "overlap_done")
    isfile(file) || return 0
    return parse(Int, strip(read(file, String)))
end

function _ts_dist_write_overlap_done!(meta_dir, r)
    file = joinpath(meta_dir, "overlap_done")
    tmp = "$(file).tmp.$(getpid())"
    write(tmp, string(r))
    return mv(tmp, file; force=true)
end

# block until the master's overlaps have advanced far enough that writing
# slice r stays within the disk budget: overlap_done >= r - backlog
function _ts_dist_wait_throttle(meta_dir, r, backlog, s; poll_s = 30.0)
    need = r - backlog
    done = _ts_dist_overlap_done(meta_dir)
    done >= need && return nothing
    println("[state $s] slice $r evolved; waiting for overlap progress $need ",
            "(now $done) to bound disk usage")
    flush(stdout)
    while done < need
        sleep(poll_s)
        done = _ts_dist_overlap_done(meta_dir)
    end
    return nothing
end

# ---- worker tasks ----

# print the real error on the worker side before it comes back to the
# master wrapped into a RemoteException
function _ts_dist_guarded(f, s, p)
    try
        return f(s, p)
    catch err
        println(stdout, "[state $s] FAILED on worker $(myid()):")
        showerror(stdout, err, catch_backtrace())
        println(stdout)
        flush(stdout)
        rethrow()
    end
end

# phase-1 task: the charged initial state of state s, checkpointed as
# slice 0. This is the memory peak of the whole run — the master throttles
# how many of these compute concurrently per host (`charged_per_node`);
# cache hits skip the queue entirely.
function _ts_dist_charged_state(s, p)
    isfile(_ts_dist_state_file(p.workdir, 0, s)) && return (s = s, origin = :checkpoint)
    i = s <= p.L ? s : s - p.L
    cache_file = p.chargedmps_path === nothing ? nothing :
                 _charged_cache_file(p.chargedmps_path, p.cache_tag, p.cache_offset + i)
    ψ = cache_file === nothing ? nothing : _charged_cache_load(cache_file)
    origin = :cache
    if ψ === nothing
        origin = :computed
        if p.verbose
            println("[state $s] computing chargedMPS on worker ", myid(),
                    " @ ", Base.Libc.gethostname())
            flush(stdout)
        end
        t0 = time_ns()
        ψ = complex(chargedMPS(p.op, p.gs, i, p.approxalg; disk = p.disk))
        cache_file === nothing || _charged_cache_save(cache_file, ψ, i, p.approxalg)
        if p.verbose
            println("[state $s] chargedMPS done on worker ", myid(), " | ",
                    _ts_ns_str(time_ns() - t0), " | RSS ",
                    _format_gib(_gc_stamp().rss))
            flush(stdout)
        end
    end
    _ts_dist_save_state(p.workdir, 0, s, ψ)
    return (s = s, origin = origin)
end

# phase-2 task: evolve state s (s <= L: bra of site i = s, dt < 0; s > L:
# ket of site i = s - L, dt > 0) through all remaining record slices,
# checkpointing to disk after each
function _ts_dist_state_evolve(s, p)
    L = p.L
    i = s <= L ? s : s - L
    family = s <= L ? "bra" : "ket"
    dtsign = s <= L ? -1 : +1
    nrec = length(p.record_indices)

    r_begin = _ts_dist_state_progress(p.workdir, nrec, s)
    r_begin == nrec && return (s = s, resumed_from = r_begin)

    t_start = time_ns()
    # slice 0 is the charged initial state written by phase 1
    ψ = _ts_dist_load_state(p.workdir, r_begin, s)
    if p.verbose
        origin = r_begin == 0 ? "charged checkpoint" :
                                "resume from slice $r_begin checkpoint"
        println("[state $s] $family $i/$L: start on worker ", myid(), " @ ",
                Base.Libc.gethostname(), " | $origin")
        flush(stdout)
    end
    envs = HalfFiniteEnvironments(ψ, p.H; disk = p.disk)

    r0 = r_begin
    # slice 1 is the initial state itself when the record window starts at
    # the first grid point (same convention as the threaded driver)
    if r0 == 0 && p.record_indices[1] == 1
        _ts_dist_save_state(p.workdir, 1, s, ψ)
        r0 = 1
    end

    gc_last = _gc_stamp()
    t_last = time_ns()
    for r in (r0 + 1):nrec
        k_from = r == 1 ? 1 : p.record_indices[r - 1]
        k_to = p.record_indices[r]
        for k in (k_from + 1):k_to
            alg = k > p.n ? p.tdvp1 : p.tdvp2
            dτ = (p.times[k] - p.times[k - 1]) / 2
            ψ, envs = fast_timestep!(ψ, p.H, 0, dtsign * dτ, alg, envs)
        end
        _ts_dist_wait_throttle(p.meta_dir, r, p.backlog, s)
        _ts_dist_save_state(p.workdir, r, s, ψ)
        if p.verbose && (r == 2 || r == nrec || r % 25 == 0 ||
                         (k_from <= p.n && p.n < k_to))
            gc_now = _gc_stamp()
            wall = time_ns() - t_last
            gc = gc_now.time - gc_last.time
            println("[state $s] $family $i/$L: slice $r/$nrec | t = $(p.times[k_to])",
                    " | maxD = ", _ts_max_bonddim((ψ,)),
                    " | wall ", _ts_ns_str(wall),
                    " | GC ", @sprintf("%.1f%%", 100 * gc / max(wall, 1)),
                    " | RSS ", _format_gib(gc_now.rss))
            flush(stdout)
            gc_last = gc_now
            t_last = time_ns()
        end
    end
    if p.verbose
        println("[state $s] $family $i/$L: all $nrec slices checkpointed | total ",
                _ts_ns_str(time_ns() - t_start))
        flush(stdout)
    end
    return (s = s, resumed_from = r_begin)
end

# ---- master-side scheduler: deterministic host placement ----

# worker pid => hostname, grouped into host => sorted pids; this is the
# only placement information the driver ever consults, so it works with
# workers launched by any cluster manager
function _ts_dist_host_map(ws)
    groups = Dict{String,Vector{Int}}()
    for w in ws
        h = remotecall_fetch(Base.Libc.gethostname, w)
        push!(get!(groups, h, Int[]), w)
    end
    foreach(sort!, values(groups))
    return groups
end

# state => host: round-robin with per-host quotas proportional to worker
# count (largest remainder), so each worker gets at most one state whenever
# nworkers >= nstates, no matter how the cluster manager ordered the workers
function _ts_dist_host_assignment(host_workers, nstates)
    hosts = sort!(collect(keys(host_workers)))
    W = sum(length(v) for v in values(host_workers))
    exact = [length(host_workers[h]) * nstates / W for h in hosts]
    quota = floor.(Int, exact)
    leftover = nstates - sum(quota)
    byfrac = sortperm(exact .- quota; rev = true)
    for k in 1:leftover
        quota[byfrac[k]] += 1
    end
    remaining = Dict(h => q for (h, q) in zip(hosts, quota))
    assignment = Vector{String}(undef, nstates)
    hi = 1
    for s in 1:nstates
        while remaining[hosts[hi]] == 0
            hi = mod1(hi + 1, length(hosts))
        end
        assignment[s] = hosts[hi]
        remaining[hosts[hi]] -= 1
        hi = mod1(hi + 1, length(hosts))
    end
    return assignment
end

# did this remotecall fail because the worker process died?
_ts_dist_worker_died(err) =
    err isa ProcessExitedException ||
    (err isa RemoteException &&
     (err.captured.ex isa ProcessExitedException ||
      (err.captured.ex isa RemoteException &&
       err.captured.ex.captured.ex isa ProcessExitedException)))

# run `taskfn(s)` for every state of one host on that host's workers, at
# most `cap` in flight; a dead worker is replaced by a spare of the same
# host and its state re-queued (every task is idempotent via checkpoints);
# a genuine task error aborts the whole run
function _ts_dist_host_run(label, taskfn, host, states, pids, cap, verbose)
    queue = sort!(copy(states))
    cap = min(cap, length(pids))
    free = collect(pids[1:cap])
    spares = collect(pids[(cap + 1):end])
    inflight = Dict{Int,Int}()   # worker => state
    done_ch = Channel{Tuple{Int,Int,Bool,Any}}(typemax(Int))
    results = Dict{Int,Any}()
    ndone = 0
    while ndone < length(states)
        while !isempty(queue) && !isempty(free)
            w = pop!(free)
            s = popfirst!(queue)
            inflight[w] = s
            # fresh copies per task: the take! variables below are named
            # wd/sd precisely so that `w`/`s` stay local to this inner loop,
            # and the `let` pins the values even if an outer `w`/`s` ever
            # comes back — capturing a shared binding would send every task
            # of a whole dispatch batch to the same worker with the same state
            @async let w = w, s = s
                ok, res = try
                    (true, remotecall_fetch(taskfn, w, s))
                catch err
                    (false, err)
                end
                put!(done_ch, (w, s, ok, res))
            end
        end
        isempty(inflight) && throw(ErrorException(
            "[$label] host $host has no live workers left but " *
            "$(length(states) - ndone) states are unfinished — resubmit the " *
            "job; everything resumes from checkpoints"))
        wd, sd, ok, res = take!(done_ch)
        delete!(inflight, wd)
        if ok
            results[sd] = res
            ndone += 1
            push!(free, wd)
            if verbose
                note = res isa NamedTuple && haskey(res, :origin) ?
                       " | " * string(res.origin) : ""
                println("[$label] state $sd done on worker $wd @ $host ",
                        "($ndone/$(length(states)) on this host)$note")
                flush(stdout)
            end
        elseif _ts_dist_worker_died(res)
            if verbose
                println("[$label] worker $wd @ $host died during state $sd; ",
                        "re-queued (it resumes from its last checkpoint)")
                flush(stdout)
            end
            pushfirst!(queue, sd)
            isempty(spares) || push!(free, pop!(spares))
            # with no spare left, the slot is lost; the surviving inflight
            # workers free up and drain the queue
        else
            throw(ErrorException(
                "[$label] state $sd on worker $wd @ $host failed: $res"))
        end
    end
    length(results) == length(states) || throw(ErrorException(
        "[$label] host $host recorded $(length(results)) distinct states " *
        "out of $(length(states)) — internal scheduler error"))
    return results
end

# fan `states` out over the hosts according to `assignment`; returns one
# Task per host — collect with `_ts_dist_collect`
function _ts_dist_schedule(label, taskfn, assignment, host_workers,
                           cap_per_host, states, verbose)
    host_states = Dict(h => Int[] for h in keys(host_workers))
    for s in states
        push!(host_states[assignment[s]], s)
    end
    tasks = Task[]
    for h in sort!(collect(keys(host_workers)))
        isempty(host_states[h]) && continue
        cap = cap_per_host === nothing ? length(host_workers[h]) :
              min(cap_per_host, length(host_workers[h]))
        push!(tasks, @async _ts_dist_host_run(label, taskfn, h, host_states[h],
                                              host_workers[h], cap, verbose))
    end
    return tasks
end

function _ts_dist_collect(tasks)
    results = Dict{Int,Any}()
    for t in tasks
        merge!(results, fetch(t))
    end
    return results
end

# ---- master: overlaps, gf file, slice lifecycle ----

# L × L Gram matrix of one slice, loaded in B × B tiles so at most 2B states
# are resident on the master; the per-pair contraction is identical to the
# threaded driver's `_ts_overlap!`
function _ts_dist_overlap_slice(workdir, r, L, B)
    dest = zeros(ComplexF64, L, L)
    for ib in 1:B:L
        ir = ib:min(ib + B - 1, L)
        bras = [_ts_dist_load_state(workdir, r, s) for s in ir]
        for jb in 1:B:L
            jr = jb:min(jb + B - 1, L)
            kets = [_ts_dist_load_state(workdir, r, L + s) for s in jr]
            tile = view(dest, ir, jr)
            Threads.@threads :dynamic for idx in CartesianIndices(tile)
                tile[idx] = dot(bras[idx[1]], kets[idx[2]])
            end
        end
    end
    return dest
end

# highest consecutive overlap slice present as an atomic meta file
function _ts_dist_gf_prefix(meta_dir, nrec)
    r = 0
    while r < nrec && isfile(_ts_dist_gf_slice_file(meta_dir, r + 1))
        r += 1
    end
    return r
end

_ts_dist_load_gf_slice(meta_dir, r) =
    jldopen(_ts_dist_gf_slice_file(meta_dir, r), "r") do f
        read(f, "pro")
    end

# save one assembled slice: atomically under meta/ (the resume source of
# truth), then appended to the user-facing gf file
function _ts_dist_save_gf_slice!(meta_dir, filename, r, slice, save)
    file = _ts_dist_gf_slice_file(meta_dir, r)
    tmp = "$(file).tmp.$(getpid())"
    jldopen(tmp, "w") do f
        f["pro"] = slice
    end
    mv(tmp, file; force=true)
    save && jldopen(filename, "a") do f
        f["pro_$r"] = slice
    end
    return nothing
end

# the user-facing gf file carries readable metadata
function _ts_dist_gf_file_healthy(filename)
    isfile(filename) || return false
    try
        jldopen(filename, "r") do f
            haskey(f, "gsenergy") || error("missing metadata")
        end
        return true
    catch
        return false
    end
end

# make sure the user-facing gf file exists and carries all slices already
# present under meta/ (a kill mid-append can leave it torn, stale, or a few
# slices behind the atomic meta files)
function _ts_dist_sync_gf_file!(filename, meta_dir, prefix, times, record_indices,
                                block, gsenergy, save)
    save || return nothing
    if _ts_dist_gf_file_healthy(filename)
        # top up from meta if the file lags the atomic per-slice records
        last_present, has_pro1 = jldopen(filename, "r") do f
            r = 0
            while r < prefix && haskey(f, "pro_$(r + 1)")
                r += 1
            end
            (r, haskey(f, "pro_1"))
        end
        prefix == 0 && has_pro1 && throw(ArgumentError(
            "$filename already contains slices from a previous (non-distributed?) " *
            "run but workdir holds no overlap records — pick another gf_path " *
            "or delete the file"))
        last_present >= prefix && return nothing
        jldopen(filename, "a") do f
            for r in (last_present + 1):prefix
                f["pro_$r"] = _ts_dist_load_gf_slice(meta_dir, r)
            end
        end
        return nothing
    else
        isfile(filename) &&
            mv(filename, "$(filename).corrupt.$(round(Int, time()))"; force = true)
        gsenergy === nothing && error("internal: gsenergy needed to recreate $filename")
        jldopen(filename, "w") do f
            f["times"] = times
            f["record_indices"] = record_indices
            f["block"] = String(block)
            f["gsenergy"] = gsenergy
            for r in 1:prefix
                f["pro_$r"] = _ts_dist_load_gf_slice(meta_dir, r)
            end
        end
        return nothing
    end
end

# self-consistency of the progress directory across restarts (NOT a physics
# validation: model / gs / op stay human-managed like the chargedMPS cache)
function _ts_dist_check_config!(meta_dir, L, block, times, record_indices, n)
    file = joinpath(meta_dir, "config.jld2")
    if isfile(file)
        cfg = load(file)
        ok = get(cfg, "format", 0) == 1 && cfg["L"] == L &&
             cfg["block"] == String(block) && cfg["times"] == collect(times) &&
             cfg["record_indices"] == record_indices && cfg["n"] == n
        ok || throw(ArgumentError(
            "workdir already contains progress for a different configuration " *
            "(block/times/record_indices/n/L mismatch): $meta_dir — use a " *
            "fresh workdir or delete the old one"))
    else
        jldopen(file, "w") do f
            f["format"] = 1
            f["L"] = L
            f["block"] = String(block)
            f["times"] = collect(times)
            f["record_indices"] = record_indices
            f["n"] = n
        end
    end
    return nothing
end

# poll until all checkpoints of slice r exist; surface scheduler failures
# instead of waiting forever
function _ts_dist_wait_slice(workdir, r, nstates, tasks; poll_s = 30.0)
    files = [_ts_dist_state_file(workdir, r, s) for s in 1:nstates]
    npresent = count(isfile, files)
    npresent == nstates && return nothing
    t0 = time_ns()
    last_log = t0
    while true
        for t in tasks
            istaskfailed(t) && wait(t)   # rethrows the scheduler error
        end
        if all(istaskdone, tasks)
            missing = findall(!isfile, files)
            throw(ErrorException(
                "all evolution tasks finished but slice $r is incomplete " *
                "($npresent/$nstates; missing states $missing) — the " *
                "checkpoints under $workdir are inconsistent, investigate"))
        end
        sleep(poll_s)
        npresent = count(isfile, files)
        npresent == nstates && return nothing
        if time_ns() - last_log > 900e9
            println("[dist] waiting for slice $r: $npresent/$nstates checkpoints, ",
                    "elapsed ", _ts_ns_str(time_ns() - t0))
            flush(stdout)
            last_log = time_ns()
        end
    end
end

function _ts_dist_dcorrelator(gs, H, op;
                              block, times, verbose, save, gf_path,
                              chargedmps_path, disk, record_indices, n,
                              approxalg, tdvp1, tdvp2, workdir, backlog,
                              overlap_block, keep_final_states, charged_per_node)
    isempty(workers()) && throw(ArgumentError(
        "mode = :distributed needs worker processes but workers() is empty. " *
        "Add them in your script before calling: `addprocs(N)` on one node, " *
        "or `using ClusterManagers; addprocs(SlurmManager(N); " *
        "exeflags = \"--project=<env> -t1\")` for SLURM multi-node. The " *
        "package never launches workers itself."))
    workdir === nothing && throw(ArgumentError(
        "mode = :distributed requires an explicit `workdir` on fast shared " *
        "storage (a few hundred GB of state checkpoints pass through it)"))
    backlog >= 1 || throw(ArgumentError("backlog must be >= 1"))
    overlap_block >= 1 || throw(ArgumentError("overlap_block must be >= 1"))
    charged_per_node === nothing || charged_per_node >= 1 ||
        throw(ArgumentError("charged_per_node must be >= 1"))

    L = length(H)
    record_indices, record_first, record_last =
        _dcorrelator_record_window(times, record_indices)
    nrec = length(record_indices)
    save && !isdir(gf_path) && mkpath(gf_path)
    mkpath(workdir)
    meta_dir = _ts_dist_meta_dir(workdir)
    mkpath(meta_dir)
    _ts_dist_check_config!(meta_dir, L, block, times, record_indices, n)
    foreach(r -> mkpath(_ts_dist_slice_dir(workdir, r)), 0:nrec)
    chargedmps_path === nothing || mkpath(chargedmps_path)

    filename = joinpath(gf_path,
                        "gf_split_start=$(times[record_first])_end=$(times[record_last])" *
                        "_block=$(block).jld2")
    prefix = _ts_dist_gf_prefix(meta_dir, nrec)
    if prefix == nrec
        verbose && println("[dist] all $nrec overlap slices already on disk; assembling")
        gf = zeros(ComplexF64, L, L, nrec)
        for r in 1:nrec
            gf[:, :, r] = _ts_dist_load_gf_slice(meta_dir, r)
        end
        if save
            # nothing = the file is healthy, so sync only tops up missing
            # slices from meta/ and never touches the metadata
            gsenergy_sync = _ts_dist_gf_file_healthy(filename) ? nothing :
                            expectation_value(gs, H)
            _ts_dist_sync_gf_file!(filename, meta_dir, prefix, times, record_indices,
                                   block, gsenergy_sync, save)
        end
        return gf
    end
    # stale slices from before the last saved overlap are useless; drop them
    # (the slice AT `prefix` stays — it is the workers' restore point)
    for r in 1:(prefix - 1)
        isdir(_ts_dist_slice_dir(workdir, r)) &&
            rm(_ts_dist_slice_dir(workdir, r); recursive = true, force = true)
    end

    gsenergy = expectation_value(gs, H)
    _ts_dist_sync_gf_file!(filename, meta_dir, prefix, times, record_indices,
                           block, gsenergy, save)
    _ts_dist_write_overlap_done!(meta_dir, prefix)
    verbose && prefix > 0 &&
        println("[dist] resuming: overlaps 1:$prefix already saved; workers resume ",
                "from their checkpoints in $workdir")

    # workers must see the module before the task closures can be
    # deserialized (`using` is toplevel-only syntax, hence the explicit
    # Core.eval)
    try
        for w in workers()
            remotecall_fetch(Core.eval, w, Main, :(using DynamicalCorrelators))
        end
    catch err
        throw(ArgumentError(
            "workers could not `using DynamicalCorrelators` — launch them with " *
            "the project activated, e.g. addprocs(...; exeflags = " *
            "\"--project=<env> -t1\"). Original error: $err"))
    end

    # ---- placement: discover hosts, split states proportionally ----
    ws = workers()
    host_workers = _ts_dist_host_map(ws)
    assignment = _ts_dist_host_assignment(host_workers, 2L)
    if verbose
        println("[dist] ", length(ws), " workers on ", length(host_workers),
                " host(s) for ", 2L, " states | backlog ", backlog,
                " slices | overlap tiles of ", overlap_block)
        for h in sort!(collect(keys(host_workers)))
            println("[dist]   ", h, ": ", length(host_workers[h]), " workers, ",
                    count(==(h), assignment), " states")
        end
        if length(ws) < 2L
            println("[dist] WARNING: fewer workers than states — some workers ",
                    "will evolve multiple states back-to-back and the wall ",
                    "time scales accordingly")
        end
        flush(stdout)
    end

    # ---- phase 1/2: charged initial states (the memory peak of the whole
    # run — strictly before any evolution, throttled per host) ----
    pending1 = [s for s in 1:(2L) if !isfile(_ts_dist_state_file(workdir, 0, s)) &&
                _ts_dist_state_progress(workdir, nrec, s) == 0]
    if isempty(pending1)
        verbose && println("[dist] phase 1/2: all charged states already on disk")
    else
        p1 = (op = op, gs = gs, L = L, approxalg = approxalg, disk = disk,
              chargedmps_path = chargedmps_path,
              cache_tag = _charged_alg_tag(approxalg),
              cache_offset = block === :greater ? 0 : L,
              workdir = workdir, verbose = verbose)
        if verbose
            println("[dist] phase 1/2: ", length(pending1), "/", 2L,
                    " charged states",
                    charged_per_node === nothing ? "" :
                    " | cap $charged_per_node computing per host")
            flush(stdout)
        end
        t_phase1 = now()
        res1 = _ts_dist_collect(
            _ts_dist_schedule("charged",
                              s -> _ts_dist_guarded(_ts_dist_charged_state, s, p1),
                              assignment, host_workers, charged_per_node,
                              pending1, verbose))
        if verbose
            origins = [r.origin for r in values(res1)]
            println("[dist] charged states done: ", count(==(:cache), origins),
                    " from cache, ", count(==(:computed), origins), " computed, ",
                    count(==(:checkpoint), origins), " already on disk | ",
                    _short_duration(now() - t_phase1))
            flush(stdout)
        end
    end

    # ---- phase 2/2: evolutions, one state per worker ----
    p2 = (H = H, L = L, times = collect(times), record_indices = record_indices,
          n = n, tdvp1 = tdvp1, tdvp2 = tdvp2, disk = disk, workdir = workdir,
          meta_dir = meta_dir, backlog = backlog, verbose = verbose)
    verbose && (println("[dist] phase 2/2: evolving ", 2L, " states"); flush(stdout))
    tasks2 = _ts_dist_schedule("evolve",
                               s -> _ts_dist_guarded(_ts_dist_state_evolve, s, p2),
                               assignment, host_workers, nothing,
                               collect(1:(2L)), verbose)
    gf = zeros(ComplexF64, L, L, nrec)
    for r in 1:prefix
        gf[:, :, r] = _ts_dist_load_gf_slice(meta_dir, r)
    end

    wall_start = now()
    verbose && _progress_start(prefix + 1, nrec,
                               "distributed time-split correlator: $block block, $L × $L")
    for r in (prefix + 1):nrec
        slice_start = now()
        _ts_dist_wait_slice(workdir, r, 2L, tasks2)
        # slice 0 (the charged initial states) is dead weight once every
        # state has a slice-1 checkpoint — drop it to bound disk usage
        r == 1 && rm(_ts_dist_slice_dir(workdir, 0); recursive = true, force = true)
        gc0 = _gc_stamp()
        dest = _ts_dist_overlap_slice(workdir, r, L, overlap_block)
        gc1 = _gc_stamp()
        _ts_assemble!(gf, dest, r, block, gsenergy, times[record_indices[r]])
        _ts_dist_save_gf_slice!(meta_dir, filename, r, gf[:, :, r], save)
        _ts_dist_write_overlap_done!(meta_dir, r)
        # retire the slice whose overlap is saved and whose successor is
        # complete (the successor is this very slice — it stays as the new
        # restore point)
        r >= 2 && rm(_ts_dist_slice_dir(workdir, r - 1); recursive = true, force = true)
        if verbose
            _progress_step(r, nrec, "t = $(times[record_indices[r]]) | slice overlap",
                           now() - slice_start)
            println("      master: GC ", _ts_ns_str(gc1.time - gc0.time),
                    " | full sweeps ", gc1.full - gc0.full,
                    " | RSS ", _format_gib(gc1.rss))
            flush(stdout)
        end
    end
    results = _ts_dist_collect(tasks2)   # all checkpoints written; rethrows errors
    if verbose
        n_resumed = count(r -> r.resumed_from > 0, values(results))
        println("[dist] all $(2L) state tasks complete ",
                "($n_resumed resumed from checkpoints, $(2L - n_resumed) from phase 1)")
    end

    if keep_final_states
        sz = sum(s -> filesize(_ts_dist_state_file(workdir, nrec, s)), 1:(2L))
        verbose && println("[dist] final-slice states kept at ",
                           _ts_dist_slice_dir(workdir, nrec), " (", _format_gib(sz),
                           "); pass keep_final_states = false to delete them")
    else
        rm(_ts_dist_slice_dir(workdir, nrec); recursive = true, force = true)
    end
    verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"),
                       " | total duration: ", _short_duration(now() - wall_start))
    return gf
end
