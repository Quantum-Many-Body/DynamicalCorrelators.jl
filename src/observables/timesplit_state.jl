_ts_slice_dir(workdir, r) = joinpath(workdir, "slice_$(lpad(r, 4, '0'))")
_ts_state_file(workdir, r, s) =
    joinpath(_ts_slice_dir(workdir, r), "state_$(lpad(s, 3, '0')).jld2")
_ts_meta_dir(workdir) = joinpath(workdir, "meta")
_ts_config_file(workdir) = joinpath(_ts_meta_dir(workdir), "config.jld2")
_ts_gf_slice_file(workdir, r) =
    joinpath(_ts_meta_dir(workdir), "pro_$(lpad(r, 4, '0')).jld2")

function _ts_atomic_jld2(f, file)
    mkpath(dirname(file))
    tmp = tempname(dirname(file))
    try
        jldopen(tmp, "w") do io
            f(io)
        end
        mv(tmp, file; force=true)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
    return file
end

function _ts_check_config!(workdir, L, block, times, record_indices, n)
    file = _ts_config_file(workdir)
    values = (format=1, L=L, block=String(block), times=collect(times),
              record_indices=record_indices, n=n)
    if isfile(file)
        cfg = load(file)
        ok = get(cfg, "format", 0) == values.format &&
             cfg["L"] == values.L && cfg["block"] == values.block &&
             cfg["times"] == values.times &&
             cfg["record_indices"] == values.record_indices && cfg["n"] == values.n
        ok || throw(ArgumentError(
            "workdir contains a different time-split configuration: $workdir"))
    else
        _ts_atomic_jld2(file) do io
            for (key, value) in pairs(values)
                io[String(key)] = value
            end
        end
    end
    return nothing
end

function _ts_state_progress(workdir, nrec, s)
    for r in nrec:-1:0
        isfile(_ts_state_file(workdir, r, s)) && return r
    end
    return -1
end

function _ts_save_state(workdir, r, s, ψ)
    file = _ts_state_file(workdir, r, s)
    _ts_atomic_jld2(file) do io
        io["state"] = ψ
        io["slice"] = r
        io["task"] = s
    end
    return file
end

function _ts_load_state(workdir, r, s)
    return jldopen(_ts_state_file(workdir, r, s), "r") do io
        read(io, "slice") == r && read(io, "task") == s || throw(ArgumentError(
            "checkpoint metadata does not match state $s at slice $r"))
        read(io, "state")
    end
end

"""
    dcorrelator_state(gs, H, op; state, block, anomalous=nothing,
                      times=0:0.05:5.0, workdir, kwargs...) -> Int

Evolve exactly one time-split state and checkpoint every recorded slice under
`workdir`. The call is independent of every other state: it launches no worker
processes, computes no overlaps, and never waits for other jobs.

For a normal block, `state = 1:L` selects the backward-evolving bras and
`state = L+1:2L` the forward-evolving kets. For `block = :gorkov`, creation and
annihilation sources occupy `1:L` and `L+1:2L`; the four families are therefore
creation bras, annihilation bras, creation kets, and annihilation kets over
`state = 1:4L`. Pass `anomalous` exactly for `:gorkov`.

`times`, `record_indices`, `n`, and `L` are guarded by `meta/config.jld2`.
Model, ground-state, operator, and algorithm choices follow the `workdir` and
`chargedmps_path` conventions. A rerun resumes from the latest checkpoint for
this state. The returned integer is the latest completed recorded slice.

The Julia process should be started with the desired number of threads. BLAS
remains single-threaded by default; `hamiltonian_threads` controls the internal
Hamiltonian fan-out and defaults to `Threads.nthreads()`.
"""
function dcorrelator_state(gs::FiniteNormalMPS, H::MPOHamiltonian,
                           op::AbstractTensorMap;
                           state::Integer,
                           block::Symbol,
                           anomalous::Union{Nothing,AbstractTensorMap}=nothing,
                           times::AbstractVector=0:0.05:5.0,
                           workdir::AbstractString,
                           chargedmps_path::Union{Nothing,AbstractString}=nothing,
                           disk::Union{Bool,AbstractString}=false,
                           record_indices::AbstractArray=1:length(times),
                           n::Integer=3,
                           approxalg=myDMRG2(),
                           tdvp1=myTDVP1(),
                           tdvp2=myTDVP2(; trunc=truncerror(; rtol=1e-3)),
                           hamiltonian_threads::Integer=Threads.nthreads(),
                           verbose::Bool=true)
    block in (:greater, :lesser, :gorkov) || throw(ArgumentError(
        "block must be :greater, :lesser or :gorkov, got :$block"))
    (block === :gorkov) == !isnothing(anomalous) || throw(ArgumentError(
        "block = :gorkov requires anomalous; normal blocks do not accept it"))
    hamiltonian_threads >= 1 || throw(ArgumentError("hamiltonian_threads must be >= 1"))
    L = length(H)
    nstates = 2_ts_nsource(L, block)
    1 <= state <= nstates || throw(ArgumentError("state must be inside 1:$nstates"))
    record_indices, _, _ = _dcorrelator_record_window(times, record_indices)
    nrec = length(record_indices)
    _ts_check_config!(workdir, L, block, times, record_indices, n)

    spec = _ts_state_spec(state, L, block, op, anomalous)
    family = spec.bra ? "bra" : "ket"
    verbose && println("[state $state/$nstates] $family site $(spec.site) on ",
        Base.Libc.gethostname(), " | pid ", getpid(), " | threads ", Threads.nthreads())

    r_begin = _ts_state_progress(workdir, nrec, state)
    if r_begin >= nrec
        verbose && println("[state $state] all $nrec slices already checkpointed")
        return r_begin
    end

    hamcap_prev = threaded_hamiltonian_threads()
    try
        set_threaded_hamiltonian!(hamiltonian_threads)
        if r_begin < 0
            t0 = time_ns()
            ψ = chargedmps_path === nothing ? nothing : _charged_cache_load(
                _charged_cache_file(chargedmps_path, _charged_alg_tag(approxalg),
                                    spec.cache_index))
            origin = ψ === nothing ? "computed" : "cache"
            if ψ === nothing
                ψ = complex(chargedMPS(spec.source_op, gs, spec.site, approxalg; disk))
                chargedmps_path === nothing || _charged_cache_save(
                    _charged_cache_file(chargedmps_path, _charged_alg_tag(approxalg),
                                        spec.cache_index), ψ, spec.site, approxalg)
            end
            _ts_save_state(workdir, 0, state, ψ)
            r_begin = 0
            verbose && println("[state $state] charged state $origin | maxD ",
                _ts_max_bonddim((ψ,)), " | wall ", _ts_ns_str(time_ns() - t0))
        else
            ψ = _ts_load_state(workdir, r_begin, state)
            verbose && println("[state $state] resume from slice $r_begin")
        end

        if r_begin == 0 && record_indices[1] == 1
            _ts_save_state(workdir, 1, state, ψ)
            r_begin = 1
            verbose && println("[state $state] slice 1/$nrec | t = $(times[1])")
        end
        r_begin >= nrec && return nrec

        envs = HalfFiniteEnvironments(ψ, H; disk)
        for r in (r_begin + 1):nrec
            t0 = time_ns()
            gc0 = _gc_stamp()
            k_from = r == 1 ? 1 : record_indices[r - 1]
            k_to = record_indices[r]
            for k in (k_from + 1):k_to
                alg = k > n ? tdvp1 : tdvp2
                dτ = (times[k] - times[k - 1]) / 2
                ψ, envs = fast_timestep!(ψ, H, 0, spec.bra ? -dτ : dτ, alg, envs)
            end
            _ts_save_state(workdir, r, state, ψ)
            if verbose
                wall = time_ns() - t0
                gc1 = _gc_stamp()
                println("[state $state] slice $r/$nrec | t = $(times[k_to])",
                    " | maxD ", _ts_max_bonddim((ψ,)),
                    " | wall ", _ts_ns_str(wall),
                    " | GC ", @sprintf("%.1f%%", 100 * (gc1.time - gc0.time) / max(wall, 1)),
                    " | RSS ", _format_gib(gc1.rss))
                flush(stdout)
            end
        end
    finally
        set_threaded_hamiltonian!(hamcap_prev)
    end
    verbose && println("[state $state] complete: $nrec/$nrec slices")
    flush(stdout)
    return nrec
end
