function _ts_read_config(workdir)
    cfg = load(_ts_config_file(workdir))
    get(cfg, "format", 0) == 1 || throw(ArgumentError("unsupported time-split checkpoint format"))
    L = cfg["L"]
    block = Symbol(cfg["block"])
    times = cfg["times"]
    record_indices = cfg["record_indices"]
    nsource = _ts_nsource(L, block)
    return (; L, block, times, record_indices, n=cfg["n"], nsource,
            nstates=2nsource, nrec=length(record_indices))
end

function _ts_gf_prefix(workdir, nrec)
    r = 0
    while r < nrec && isfile(_ts_gf_slice_file(workdir, r + 1))
        r += 1
    end
    return r
end

_ts_slice_ready(workdir, r, nstates) =
    all(s -> isfile(_ts_state_file(workdir, r, s)), 1:nstates)

function _ts_overlap_tile!(dest, bras, kets)
    Threads.@threads :dynamic for index in CartesianIndices(dest)
        dest[index] = dot(bras[index[1]], kets[index[2]])
    end
    return dest
end

function _ts_normal_checkpoint_overlap(workdir, r, L, B)
    dest = zeros(ComplexF64, L, L)
    for ib in 1:B:L
        ir = ib:min(ib + B - 1, L)
        bras = [_ts_load_state(workdir, r, s) for s in ir]
        for jb in 1:B:L
            jr = jb:min(jb + B - 1, L)
            kets = [_ts_load_state(workdir, r, L + s) for s in jr]
            _ts_overlap_tile!(view(dest, ir, jr), bras, kets)
        end
    end
    return dest
end

function _ts_gorkov_checkpoint_overlap(workdir, r, L, B)
    dest = ntuple(_ -> zeros(ComplexF64, L, L), 4)
    for ib in 1:B:L
        ir = ib:min(ib + B - 1, L)
        for jb in 1:B:L
            jr = jb:min(jb + B - 1, L)
            creation_bras = [_ts_load_state(workdir, r, s) for s in ir]
            annihilation_bras = [_ts_load_state(workdir, r, L + s) for s in ir]
            creation_kets = [_ts_load_state(workdir, r, 2L + s) for s in jr]
            annihilation_kets = [_ts_load_state(workdir, r, 3L + s) for s in jr]
            _ts_overlap_tile!(view(dest[1], ir, jr), creation_bras, creation_kets)
            _ts_overlap_tile!(view(dest[2], ir, jr), annihilation_bras, annihilation_kets)
            _ts_overlap_tile!(view(dest[3], ir, jr), annihilation_bras, creation_kets)
            _ts_overlap_tile!(view(dest[4], ir, jr), creation_bras, annihilation_kets)
        end
    end
    return dest
end

function _ts_assemble_slice(raw, block, gsenergy, t)
    if block === :gorkov
        L = size(raw[1], 1)
        gf = zeros(ComplexF64, 2L, 2L, 1)
        _ts_gorkov_assemble!(gf, raw, 1, gsenergy, t)
    else
        L = size(raw, 1)
        gf = zeros(ComplexF64, L, L, 1)
        _ts_assemble!(gf, raw, 1, block, gsenergy, t)
    end
    return reshape(gf, size(gf, 1), size(gf, 2))
end

function _ts_save_gf_slice(workdir, r, slice)
    return _ts_atomic_jld2(_ts_gf_slice_file(workdir, r)) do io
        io["pro"] = slice
        io["slice"] = r
    end
end

function _ts_retire_states!(workdir, overlapped, nstates, nrec, retention)
    for s in 1:nstates
        latest = _ts_state_progress(workdir, nrec, s)
        for r in 0:min(overlapped, latest - retention)
            rm(_ts_state_file(workdir, r, s); force=true)
        end
    end
    for r in 0:overlapped
        dir = _ts_slice_dir(workdir, r)
        isdir(dir) && isempty(readdir(dir)) && rm(dir)
    end
    return nothing
end

function _ts_collect_gf(workdir, cfg, prefix)
    gf = zeros(ComplexF64, cfg.nsource, cfg.nsource, prefix)
    for r in 1:prefix
        gf[:, :, r] = jldopen(_ts_gf_slice_file(workdir, r), "r") do io
            read(io, "pro")
        end
    end
    return gf
end

function _ts_save_gf(workdir, gf_path, cfg, gsenergy, gf)
    mkpath(gf_path)
    filename = joinpath(gf_path,
        "gf_split_start=$(cfg.times[first(cfg.record_indices)])" *
        "_end=$(cfg.times[last(cfg.record_indices)])_block=$(cfg.block).jld2")
    _ts_atomic_jld2(filename) do io
        io["times"] = cfg.times
        io["record_indices"] = cfg.record_indices
        io["block"] = String(cfg.block)
        io["gsenergy"] = gsenergy
        for r in axes(gf, 3)
            io["pro_$r"] = gf[:, :, r]
        end
    end
    return filename
end

"""
    dcorrelator_overlap(workdir; gsenergy, gf_path=workdir, kwargs...)

Standalone overlap job for checkpoints written by [`dcorrelator_state`](@ref).
It reads `meta/config.jld2`, finds consecutive slices for which every state
checkpoint exists, computes the normal or four-block Gorkov overlaps, and saves
each assembled slice atomically under `meta/pro_NNNN.jld2`.

State jobs never wait for this function. With `once = false` (the default), the
call polls until all slices are complete, writes the combined `gf_split_...jld2`
file, and returns the full Green-function array. With `once = true`, it processes
all currently ready slices and returns only the completed prefix; a missing
configuration is then an error.

After each overlap is safely saved, old checkpoints may be deleted. `retention =
2` keeps the latest two checkpoints of each state and never removes a slice whose
overlap has not been saved. Run only one overlap job per `workdir`.

`overlap_block` bounds each loaded tile; Gorkov tiles load creation and
annihilation families together and share each loaded tile across all four blocks.
`poll_interval` is in seconds. Logs are not used to determine data readiness.
"""
function dcorrelator_overlap(workdir::AbstractString;
                             gsenergy,
                             gf_path::AbstractString=workdir,
                             save::Bool=true,
                             once::Bool=false,
                             poll_interval::Real=30.0,
                             retention::Integer=2,
                             overlap_block::Integer=8,
                             verbose::Bool=true)
    poll_interval > 0 || throw(ArgumentError("poll_interval must be positive"))
    retention >= 1 || throw(ArgumentError("retention must be >= 1"))
    overlap_block >= 1 || throw(ArgumentError("overlap_block must be >= 1"))
    while !isfile(_ts_config_file(workdir))
        once && throw(ArgumentError("no state job has initialized $workdir"))
        verbose && println("[overlap] waiting for ", _ts_config_file(workdir))
        sleep(poll_interval)
    end
    cfg = _ts_read_config(workdir)
    prefix = _ts_gf_prefix(workdir, cfg.nrec)
    last_log = time_ns()
    while true
        while prefix < cfg.nrec && _ts_slice_ready(workdir, prefix + 1, cfg.nstates)
            r = prefix + 1
            t0 = time_ns()
            raw = cfg.block === :gorkov ?
                _ts_gorkov_checkpoint_overlap(workdir, r, cfg.L, overlap_block) :
                _ts_normal_checkpoint_overlap(workdir, r, cfg.L, overlap_block)
            slice = _ts_assemble_slice(raw, cfg.block, gsenergy, cfg.times[cfg.record_indices[r]])
            _ts_save_gf_slice(workdir, r, slice)
            prefix = r
            _ts_retire_states!(workdir, prefix, cfg.nstates, cfg.nrec, retention)
            verbose && println("[overlap] slice $r/$(cfg.nrec) | t = ",
                cfg.times[cfg.record_indices[r]], " | wall ", _ts_ns_str(time_ns() - t0))
            flush(stdout)
        end
        if prefix == cfg.nrec
            gf = _ts_collect_gf(workdir, cfg, prefix)
            save && _ts_save_gf(workdir, gf_path, cfg, gsenergy, gf)
            verbose && println("[overlap] complete: $prefix/$(cfg.nrec) slices")
            return gf
        end
        once && return _ts_collect_gf(workdir, cfg, prefix)
        if verbose && time_ns() - last_log >= 900e9
            ready = count(isfile, [_ts_state_file(workdir, prefix + 1, s)
                                   for s in 1:cfg.nstates])
            println("[overlap] waiting for slice $(prefix + 1): $ready/$(cfg.nstates) checkpoints")
            flush(stdout)
            last_log = time_ns()
        end
        sleep(poll_interval)
    end
end
