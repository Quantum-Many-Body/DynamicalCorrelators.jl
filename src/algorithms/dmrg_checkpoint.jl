# Update-level (mid-sweep) checkpointing for the DMRG sweep driver.
#
# The rolling sweep checkpoint (`filename`, JLD2) only protects completed
# sweeps: a kill mid-sweep loses every update done so far, and the
# disk-backed environments on the node's scratch are orphaned because a fresh
# `HalfFiniteEnvironments` lands in new random subdirectories. The machinery
# here closes that gap:
#
#   * after every local update, the touched site/bond tensors are appended to
#     the updates directory (one file per site/bond, atomic tmp+rename),
#     followed by a tiny `meta.bin` (also atomic) recording the sweep
#     bookkeeping and WHICH environment entries are consistent with the saved
#     tensors (computed from the in-memory dependency pointers, so it can
#     never mark a stale entry as valid);
#   * the directory lives next to the environments when `disk` is a path
#     (`joinpath(disk, basename(filename) * ".updates.d")` — transient
#     scratch, e.g. bbfs scratch, where it avoids the writeback traffic of
#     the persistent side), and next to `filename` otherwise;
#   * on entry, `_dmrg_run!` looks for that directory and, if the schedule
#     matches, resumes: the state is the rolling checkpoint (sweep start)
#     overlaid with the per-update files, and surviving environment files are
#     renamed into the new manager's arrays with their dependency pointers
#     re-pointed at the loaded tensors (restoring the === staleness
#     semantics). Anything missing or mismatched falls back to the normal
#     lazy rebuild — resume is best-effort, never a correctness risk.
#
# Update-level checkpointing requires `save === true` (the rolling checkpoint
# is the base state the per-update files overlay); it is inactive for
# `save = false` and history mode (`save = [...]`). All files here are
# `Serialization` format — internal scratch, not user-facing (the rolling
# JLD2 checkpoint remains the user-facing state).

const _UPDATE_CKPT_VERSION = 1

# the context record carried through the sweep loop
mutable struct _UpdateCkptContext
    dir::String
    schedule_id::UInt
    nscheduled::Int
    gls_path::String
    grs_path::String
    base_fingerprint::Any   # refreshed at every sweep start
end

_updates_dir(filename::AbstractString, disk = false) =
    disk isa AbstractString ?
        joinpath(disk, basename(filename) * ".updates.d") :
        filename * ".updates.d"

_maybe(x) = ismissing(x) ? nothing : x
_unmaybe(x) = x === nothing ? missing : x

function _write_atomic(path::AbstractString, value)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        serialize(io, value)
    end
    mv(tmp, path; force = true)
    return nothing
end

# ---------------------------------------------------------------------------
# write side
# ---------------------------------------------------------------------------

# A cheap identity of the sweep-start state: per-bond dimensions plus the
# norm. Used on resume to decide whether the caller's ψ IS the base state
# (the common case: the script just reloaded the rolling checkpoint) or the
# base has to be read back from `filename`.
function _state_fingerprint(ψ::AbstractFiniteMPS)
    N = length(ψ)
    dims = ntuple(b -> dim(right_virtualspace(ψ, b)), N - 1)
    return (dims, norm(ψ))
end

function _fingerprint_match(fp, ψ::AbstractFiniteMPS)
    dims, nrm = fp
    dims == _state_fingerprint(ψ)[1] || return false
    return abs(nrm - norm(ψ)) <= 1.0e-10 * max(one(abs(nrm)), abs(nrm))
end

# environment consistency map WITHOUT touching the filesystem: an entry is
# valid iff its dependency pointer is alive and still identical to the
# current tensor (freed entries have their dependency cleared by
# `free_left!/free_right!`, so a missing file implies a nothing dependency;
# the converse — file missing with a live dependency — can only happen
# through external deletion and is re-checked with isfile at resume time)
function _env_validity(envs::HalfFiniteEnvironments, ψ::AbstractFiniteMPS)
    N = length(envs)
    gls = falses(N + 1)
    grs = falses(N + 1)
    gls[1] = true       # boundary: never freed, no dependency
    grs[N + 1] = true   # boundary
    for j in 1:N
        dep = envs.ldependencies[j]
        gls[j + 1] = dep !== nothing && !ismissing(ψ.ALs[j]) && dep === ψ.ALs[j]
        dep = envs.rdependencies[j]
        grs[j] = dep !== nothing && !ismissing(ψ.ARs[j]) && dep === ψ.ARs[j]
    end
    return gls, grs
end
_env_validity(::FiniteEnvironments, ::AbstractFiniteMPS) = falses(0), falses(0)

function _env_paths(envs::HalfFiniteEnvironments)
    pathof(v) = begin
        store = _disk_store(v)
        store isa SerializedElementArray ? pathname(store) : ""
    end
    return pathof(envs.GLs), pathof(envs.GRs)
end
_env_paths(::FiniteEnvironments) = "", ""

"""
    _update_ckpt_state!(ctx, ψ, envs; iter, dir, pos, done_sweep, E₀, ΔE, E_prev, ϵ_truncs, nalgs)

Write the per-update checkpoint: first the site/bond tensors that changed
since the captured `pre` snapshot (identity comparison — unchanged entries
keep the same tensor object), then the metadata atomically. `pre` is a
4-tuple of COPIES of ψ's internal vectors taken before the update
(`nothing` for the sweep-end call, which writes no tensors).
"""
function _update_ckpt_state!(
        ctx::_UpdateCkptContext, ψ, envs, pre;
        engine::Symbol, iter::Int, dir::Symbol, pos::Int, done_sweep::Bool,
        E₀::Float64, ΔE::Float64, E_prev::Float64, ϵ_truncs, nalgs::Int
    )
    dirpath = ctx.dir
    mkpath(dirpath)
    if pre !== nothing
        pre_ALs, pre_ARs, pre_ACs, pre_Cs = pre
        N = length(ψ)
        for i in 1:N
            if !(ψ.ALs[i] === pre_ALs[i]) || !(ψ.ARs[i] === pre_ARs[i]) ||
                    !(ψ.ACs[i] === pre_ACs[i])
                _write_atomic(
                    joinpath(dirpath, "site_$i.bin"),
                    (AL = _maybe(ψ.ALs[i]), AR = _maybe(ψ.ARs[i]),
                        AC = _maybe(ψ.ACs[i]))
                )
            end
        end
        for i in 1:(N + 1)
            if !(ψ.Cs[i] === pre_Cs[i])
                _write_atomic(joinpath(dirpath, "bond_$i.bin"), _maybe(ψ.Cs[i]))
            end
        end
    end
    gls_valid, grs_valid = _env_validity(envs, ψ)
    meta = (
        version = _UPDATE_CKPT_VERSION,
        N = length(ψ),
        engine = engine,
        schedule_id = ctx.schedule_id,
        base_fingerprint = ctx.base_fingerprint,
        iter = iter, dir = dir, pos = pos, done_sweep = done_sweep,
        E₀ = E₀, ΔE = ΔE, E_prev = E_prev,
        ϵ_truncs = copy(ϵ_truncs),
        nscheduled = ctx.nscheduled, nalgs = nalgs,
        gls_path = ctx.gls_path, grs_path = ctx.grs_path,
        gls_valid = gls_valid, grs_valid = grs_valid,
    )
    _write_atomic(joinpath(dirpath, "meta.bin"), meta)
    return nothing
end

# drop the per-update tensor files (the rolling checkpoint is authoritative
# from here on); meta.bin stays so a resume can still reuse environments
function _clear_update_tensors(dirpath::AbstractString)
    for f in readdir(dirpath)
        (startswith(f, "site_") || startswith(f, "bond_")) &&
            rm(joinpath(dirpath, f); force = true)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# read side
# ---------------------------------------------------------------------------

function _load_update_meta(dirpath::AbstractString)
    file = joinpath(dirpath, "meta.bin")
    isfile(file) || return nothing
    meta = deserialize(file)
    meta isa NamedTuple || return nothing
    get(meta, :version, 0) == _UPDATE_CKPT_VERSION || return nothing
    return meta
end

_meta_matches(meta, N::Int, engine::Symbol, schedule_id::Unsigned, nscheduled::Int) =
    meta.N == N && meta.engine == engine && meta.schedule_id == schedule_id &&
    meta.nscheduled == nscheduled

# copy the full state of `src` into `dst`'s storage in place, preserving the
# `dst` object identity (dmrgX! are in-place APIs: the caller's ψ must end up
# holding the result, and environment managers are built around that object)
function _copy_state!(dst::AbstractFiniteMPS, src::AbstractFiniteMPS)
    N = length(dst)
    for i in 1:N
        dst.ALs[i] = src.ALs[i]
        dst.ARs[i] = src.ARs[i]
        dst.ACs[i] = src.ACs[i]
    end
    for i in 1:(N + 1)
        dst.Cs[i] = src.Cs[i]
    end
    return dst
end

# overlay the per-update files onto the base (sweep-start) state in place
function _overlay_update_tensors!(ψ::AbstractFiniteMPS, dirpath::AbstractString)
    n = 0
    for f in readdir(dirpath)
        endswith(f, ".tmp") && continue   # torn write leftovers are ignored
        if startswith(f, "site_")
            i = parse(Int, f[6:(end - 4)])
            t = deserialize(joinpath(dirpath, f))
            ψ.ALs[i] = _unmaybe(t.AL)
            ψ.ARs[i] = _unmaybe(t.AR)
            ψ.ACs[i] = _unmaybe(t.AC)
            n += 1
        elseif startswith(f, "bond_")
            i = parse(Int, f[6:(end - 4)])
            ψ.Cs[i] = _unmaybe(deserialize(joinpath(dirpath, f)))
        end
    end
    return ψ, n
end

# rename the surviving valid entries of the recorded arrays into the new
# manager's arrays and re-point the dependency slots at the loaded tensors,
# restoring the === staleness semantics; invalid/missing entries are dropped
function _resume_disk_envs!(
        envs::HalfFiniteEnvironments, ψ::AbstractFiniteMPS,
        gls_path::String, grs_path::String, gls_valid, grs_valid
    )
    N = length(envs)
    n_l = _move_env_entries!(
        envs.GLs, gls_path, gls_valid, N,
        j -> j == 1 ? nothing : j - 1,          # GLs[j] ↔ ldependencies[j-1]
        k -> ψ.ALs[k], envs.ldependencies
    )
    n_r = _move_env_entries!(
        envs.GRs, grs_path, grs_valid, N,
        j -> j == N + 1 ? nothing : j,          # GRs[j] ↔ rdependencies[j]
        k -> ψ.ARs[k], envs.rdependencies
    )
    return n_l, n_r
end

function _move_env_entries!(arr, oldpath::String, valid, N, dep_index, tensor_at, deps)
    store = _disk_store(arr)
    store isa SerializedElementArray || return 0
    (isempty(oldpath) || !isdir(oldpath)) && return 0
    oldpath == pathname(store) && return 0
    reused = 0
    for j in 1:(N + 1)
        oldf = joinpath(oldpath, "$j.bin")
        isfile(oldf) || continue
        if j <= length(valid) && valid[j]
            mv(oldf, filename(store, j); force = true)
            k = dep_index(j)
            k === nothing || (deps[k] = tensor_at(k))
            reused += 1
        else
            rm(oldf; force = true)
        end
    end
    rm(oldpath; recursive = true, force = true)
    return reused
end

"""
    _resume_update_checkpoint(filename, ψ, algs, envs, schedule_id, verbose) -> NamedTuple or nothing

Detect and consume a mid-sweep checkpoint. On success returns the resume
state (`ψ` with the saved tensors overlaid, the loop bookkeeping, and the
resume point); on any mismatch or missing piece the directory is discarded
and `nothing` is returned, i.e. a fresh start.
"""
function _resume_update_checkpoint(
        filename::String, ψ::AbstractFiniteMPS, algs, envs,
        schedule_id::Unsigned, verbose, disk = false
    )
    dirpath = _updates_dir(filename, disk)
    meta = _load_update_meta(dirpath)
    meta === nothing && return nothing
    N = length(ψ)
    iter = meta.iter
    1 <= iter || return _discard_update_checkpoint(dirpath, "corrupt sweep index")
    engine = nameof(typeof(algs[min(iter, length(algs))]))
    if !_meta_matches(meta, N, engine, schedule_id, length(algs))
        return _discard_update_checkpoint(
            dirpath, "schedule mismatch (the directory belongs to a different run)"
        )
    end
    # tail-sweep reconstruction: the crashed run had already appended its
    # continuation copies, so regrow the schedule to the recorded length
    while length(algs) < meta.nalgs
        push!(algs, algs[end])
    end

    # base state: mid-sweep → rolling checkpoint overlaid with the tensor
    # files; done_sweep → the rolling checkpoint alone. Either way the state
    # is written INTO the caller's ψ object (dmrgX! are in-place APIs).
    if meta.done_sweep
        isfile(filename) ||
            return _discard_update_checkpoint(dirpath, "rolling checkpoint missing")
        _copy_state!(ψ, load(filename, "ψ"))
        ntensors = 0
    else
        if _fingerprint_match(meta.base_fingerprint, ψ)
            # the caller's ψ IS the sweep-start state (the usual rerun)
        elseif isfile(filename)
            ψbase = load(filename, "ψ")
            _fingerprint_match(meta.base_fingerprint, ψbase) ||
                return _discard_update_checkpoint(dirpath, "base state mismatch")
            _copy_state!(ψ, ψbase)
        else
            return _discard_update_checkpoint(dirpath, "base state unavailable")
        end
        ψ, ntensors = _overlay_update_tensors!(ψ, dirpath)
    end

    n_l = n_r = 0
    if envs isa HalfFiniteEnvironments
        n_l, n_r = _resume_disk_envs!(
            envs, ψ, meta.gls_path, meta.grs_path, meta.gls_valid, meta.grs_valid
        )
    end
    Int(verbose) > 0 && println(
        "  update-level checkpoint found: resuming sweep $(meta.iter)",
        meta.done_sweep ? " (sweep complete)" : " from $(meta.dir) position $(meta.pos)",
        " | tensors replayed: $ntensors | environments reused: $(n_l) left, $(n_r) right"
    )
    return (
        ψ = ψ, iter = iter, E_prev = meta.E_prev, ϵ_truncs = meta.ϵ_truncs,
        done_sweep = meta.done_sweep, E₀ = meta.E₀, ΔE = meta.ΔE,
        base_fingerprint = meta.base_fingerprint,
        resume_point = meta.done_sweep ? nothing : (dir = meta.dir, pos = meta.pos),
    )
end

function _discard_update_checkpoint(dirpath::AbstractString, reason::String)
    rm(dirpath; recursive = true, force = true)
    @info "update-level checkpoint discarded ($reason)"
    return nothing
end

function _update_ckpt_context(
        filename::String, nscheduled::Int, envs, schedule_id::Unsigned, disk = false
    )
    gls_path, grs_path = _env_paths(envs)
    return _UpdateCkptContext(
        _updates_dir(filename, disk), schedule_id, nscheduled,
        gls_path, grs_path, nothing
    )
end
