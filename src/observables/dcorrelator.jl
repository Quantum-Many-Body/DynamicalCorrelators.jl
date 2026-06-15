"""
    evolve_mps(H::MPOHamiltonian, ts::AbstractVector,
               rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H)); kwargs...)

Evolve `rho_mps` through the time grid `ts` under a time-independent
Hamiltonian `H`.

The first `n` recorded steps use `tdvp2` and later steps use `tdvp1`. Saved
states are written to `filename` for every index in `save_id` with keys
`"t=\$(ts[i])"`.
"""
function evolve_mps(H::MPOHamiltonian, ts::AbstractVector, rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H)); 
                    filename::String="default_expiHt_ψ.jld2", 
                    save_id::AbstractArray=[length(ts),], 
                    verbose::Bool=true, 
                    n::Integer=3, 
                    trscheme=truncerror(;rtol=1e-3),
                    tdvp1 = myTDVP,
                    tdvp2 = myTDVP2(trscheme)
                    )
    start_time, record_start = now(), now()
    verbose && println("[1/$(length(ts))] t = $(ts[1]) ", " | Started:", Dates.format(start_time, "d.u yyyy HH:MM"))
    flush(stdout)
    envs = environments(rho_mps, H)
    jldopen(filename, "w") do f
        f["ts"] = ts
        if 1 in save_id
            f["t=$(ts[1])"] = rho_mps
        end
    end
    for i in 2:length(ts)
        alg = i > n ? tdvp1 : tdvp2
        rho_mps, envs = timestep(rho_mps, H, 0, ts[i]-ts[i-1], alg, envs)
        current_time = now()
        verbose && println("[$i/$(length(ts))] t = $(ts[i]) ", " | duration:", Dates.canonicalize(current_time-start_time))
        flush(stdout)
        jldopen(filename, "a") do f
            if i in save_id
                f["t=$(ts[i])"] = rho_mps
            end
        end
        start_time = current_time
    end
    verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    return rho_mps
end

"""
    evolve_mps(H::Function, ts::AbstractVector, mus::AbstractVector,
               rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H(mus[1]))); kwargs...)

Evolve `rho_mps` through `ts` under a time-dependent Hamiltonian `H(mu)`.

At step `i`, the Hamiltonian is rebuilt as `H(mus[i])`. Saved states and the
corresponding parameter values are written to `filename` with keys
`"t=\$(ts[i])"` and `"mu_t=\$(ts[i])"`.
"""
function evolve_mps(H::Function, ts::AbstractVector, mus::AbstractVector, rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H(mus[1]))); 
                    filename::String="default_expiHt_ψ.jld2", 
                    save_id::AbstractArray=[length(ts),], 
                    verbose::Bool=true, 
                    n::Integer=3, 
                    trscheme=truncerror(;rtol=1e-3),
                    tdvp1 = myTDVP,
                    tdvp2 = myTDVP2(trscheme)
                    )
    start_time, record_start = now(), now()
    verbose && println("[1/$(length(ts))] t = $(ts[1]) ", " | Started:", Dates.format(start_time, "d.u yyyy HH:MM"))
    flush(stdout)
    envs = environments(rho_mps, H(mus[1]))
    jldopen(filename, "w") do f
        f["ts"] = ts
        if 1 in save_id
            f["t=$(ts[1])"] = rho_mps
            f["mu_t=$(ts[1])"] = mus[1]
        end
    end
    for i in 2:length(ts)
        alg = i > n ? tdvp1 : tdvp2
        rho_mps, envs = timestep(rho_mps, H(mus[i]), 0, ts[i]-ts[i-1], alg, envs)
        current_time = now()
        verbose && println("[$i/$(length(ts))] t = $(ts[i]) ", " | duration:", Dates.canonicalize(current_time-start_time))
        flush(stdout)
        jldopen(filename, "a") do f
            if i in save_id
                f["t=$(ts[i])"] = rho_mps
                f["mu_t=$(ts[i])"] = mus[i]
            end
        end
        start_time = current_time
    end
    verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    return rho_mps
end

function _dcorrelator_record_window(times, record_indices)
    record_indices = collect(record_indices)
    isempty(record_indices) && throw(ArgumentError("record_indices cannot be empty"))
    record_indices == collect(record_indices[1]:record_indices[end]) ||
        throw(ArgumentError("record_indices must be continuous"))
    1 <= record_indices[1] <= record_indices[end] <= length(times) ||
        throw(ArgumentError("record_indices must be inside 1:$(length(times))"))
    return record_indices, record_indices[1], record_indices[end]
end

function _dcorrelator_site(id::Integer, L::Integer; doubled::Bool=true)
    upper = doubled ? 2L : L
    1 <= id <= upper || throw(ArgumentError("id must be inside 1:$(upper)"))
    return id <= L ? id : id - L
end

function _dcorrelator_load_complete!(gf, filename::String, nrecords::Integer; verbose::Bool=false)
    isfile(filename) || return false
    gfb = load(filename)
    iscomplete = all("pro_$(r)" in keys(gfb) for r in 1:nrecords)
    if iscomplete
        for r in 1:nrecords
            gf[:, r] = gfb["pro_$(r)"]
        end
        verbose && println("$(basename(filename)) has existed!")
        verbose && flush(stdout)
        return true
    end
    @warn "$(filename) is incomplete; recomputing it"
    return false
end

"""
    dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian,
                op::AbstractTensorMap, id::Integer; kwargs...)

Compute a zero-temperature dynamical correlation from one source site.

`id` selects the source channel. Values `1:length(H)` use the forward
operator channel, while `length(H)+1:2length(H)` use the conjugated channel.
Only `record_indices` are returned, and the evolution stops at the last
requested record. This single-source method uses ordinary arrays and does not
start a distributed `SharedArray` calculation.
"""
function dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap, id::Integer;
                    verbose=true,
                    gf_path::String="./",
                    times::AbstractRange=0:0.05:5.0,
                    record_indices::AbstractArray=1:length(times),
                    n::Integer=3,
                    trscheme=truncerror(;rtol=1e-3),
                    approxalg = myDMRG2(),
                    tdvp1 = myTDVP,
                    tdvp2 = myTDVP2(trscheme)
                    )
    !isdir(gf_path) && mkpath(gf_path)
    L = length(H)
    idx = _dcorrelator_site(id, L)
    record_indices, record_first, record_last = _dcorrelator_record_window(times, record_indices)
    gsenergy = expectation_value(gs, H)
    gf = zeros(ComplexF64, L, length(record_indices))
    filename = joinpath(gf_path, "gf_start=$(times[record_first])_end=$(times[record_last])_id=$(id).jld2")

    _dcorrelator_load_complete!(gf, filename, length(record_indices); verbose=verbose) && return gf

    ket = chargedMPS(op, gs, idx, approxalg)
    start_time, wall_start = now(), now()
    jldopen(filename, "w") do f
        f["times"] = times
        f["record_indices"] = record_indices
        f["id"] = id
        if record_first == 1
            phase = id <= L ? exp(im*gsenergy*times[1]) : exp(-im*gsenergy*times[1])
            sd = sweep_dot(gs, op, ket)
            gf[:, 1] = id <= L ? -im * phase .* sd : -im * phase .* conj.(sd)
            f["pro_1"] = gf[:, 1]
        end
    end
    verbose && println("[1/$(length(times))] Started: time evolves 0 of ket$(id) ", Dates.format(start_time, "d.u yyyy HH:MM"))
    verbose && flush(stdout)

    envs = environments(ket, H)
    for k in 2:record_last
        alg = k > n ? tdvp1 : tdvp2
        ket, envs = timestep(ket, H, 0, times[k] - times[k - 1], alg, envs)
        current_time = now()
        verbose && println("[$(k)/$(length(times))] time evolves $(times[k]) of ket$(id) ",
            " | duration:", Dates.canonicalize(current_time - start_time))
        verbose && flush(stdout)
        if k >= record_first
            r = k - record_first + 1
            phase = id <= L ? exp(im*gsenergy*times[k]) : exp(-im*gsenergy*times[k])
            sd = sweep_dot(gs, op, ket)
            gf[:, r] = id <= L ? -im * phase .* sd : -im * phase .* conj.(sd)
            jldopen(filename, "a") do f
                f["pro_$(r)"] = gf[:, r]
            end
        end
        start_time = current_time
    end

    ket = nothing
    envs = nothing
    GC.gc()
    verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now() - wall_start))
    return gf
end

"""
    dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian,
                op::AbstractTensorMap, indices::AbstractArray; kwargs...)

Compute zero-temperature dynamical correlations for many source channels.

The source channels in `indices` are evaluated in parallel with Distributed and
stored in a `SharedArray`. The result has size
`(length(H), length(indices), length(record_indices))`. Completed per-source
JLD2 files are reused; incomplete files are recomputed.
"""
function dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true,
                    gf_path::String="./",
                    times::AbstractRange=0:0.05:5.0,
                    record_indices::AbstractArray=1:length(times),
                    n::Integer=3,
                    trscheme=truncerror(;rtol=1e-3),
                    approxalg = myDMRG2(),
                    tdvp1 = myTDVP,
                    tdvp2 = myTDVP2(trscheme)
                    )
    !isdir(gf_path) && mkdir(gf_path)
    record_indices = collect(record_indices)
    isempty(record_indices) && throw(ArgumentError("record_indices cannot be empty"))
    record_indices == collect(record_indices[1]:record_indices[end]) ||
        throw(ArgumentError("record_indices must be continuous"))
    1 <= record_indices[1] <= record_indices[end] <= length(times) ||
        throw(ArgumentError("record_indices must be inside 1:$(length(times))"))
    record_first, record_last = record_indices[1], record_indices[end]
    gsenergy = expectation_value(gs, H)
    gf = SharedArray{ComplexF64, 3}(length(H), length(indices), length(record_indices))
    @sync @distributed for d in eachindex(indices)
        id = indices[d]
        idx = id <= length(H) ? id : (id - length(H))
        filename = joinpath(gf_path, "gf_start=$(times[record_first])_end=$(times[record_last])_id=$(id).jld2")
        if isfile(filename)
            gfb = load(filename)
            iscomplete = all("pro_$(r)" in keys(gfb) for r in 1:length(record_indices))
            if iscomplete
                for r in 1:length(record_indices)
                    gf[:,d,r] = gfb["pro_$(r)"]
                end
                verbose && println("gf_start=$(times[record_first])_end=$(times[record_last])_id=$(id).jld2 has existed!")
                flush(stdout)
                continue
            end
            @warn "$(filename) is incomplete; recomputing it"
        end
        ket = chargedMPS(op, gs, idx, approxalg)
        start_time, wall_start = now(), now()
        jldopen(filename, "w") do f
            if record_first == 1
                phase = id <= length(H) ? exp(im*gsenergy*times[1]) : exp(-im*gsenergy*times[1])
                sd = sweep_dot(gs, op, ket)
                gf[:,d,1] = id <= length(H) ? -im * phase .* sd : -im * phase .* conj.(sd)
                f["pro_1"] = gf[:,d,1]
            end
        end
        verbose && println("[1/$(length(times))] Started: time evolves 0 of ket$(id) ", Dates.format(start_time, "d.u yyyy HH:MM"))
        flush(stdout)
        envs = environments(ket, H)
        for k in 2:record_last
            alg = k > n ? tdvp1 : tdvp2
            ket, envs = timestep(ket, H, 0, times[k]-times[k-1], alg, envs)
            if k >= record_first
                r = k - record_first + 1
                phase = id <= length(H) ? exp(im*gsenergy*times[k]) : exp(-im*gsenergy*times[k])
                sd = sweep_dot(gs, op, ket)
                gf[:,d,r] = id <= length(H) ? -im * phase .* sd : -im * phase .* conj.(sd)
                current_time = now()
                verbose && println("[$(k)/$(length(times))] time evolves $(times[k]) of ket$(id) ", " | duration:", Dates.canonicalize(current_time-start_time))
                flush(stdout)
                jldopen(filename, "a") do f
                    f["pro_$(r)"] = gf[:,d,r]
                end
            else
                current_time = now()
                verbose && println("[$(k)/$(length(times))] time evolves $(times[k]) of ket$(id) ", " | duration:", Dates.canonicalize(current_time-start_time))
                flush(stdout)
            end
            start_time = current_time
        end
        ket = nothing
        envs = nothing
        GC.gc()
        verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-wall_start))
    end
    gfs = zeros(ComplexF64, length(H), length(indices), length(record_indices))
    gfs .= gf
    return gfs
end

"""
    dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap};
            verbose=true,
            gf_path::String="./",
            times::AbstractRange=0:0.05:5.0,
            n::Integer=3,
            trscheme=truncerror(;rtol=1e-3),
            tdvp1 = myTDVP,
            tdvp2 = myTDVP2(trscheme),
            isfermion::Bool = true
            )

Compute zero-temperature correlations for a pair of source operators.

The first `length(H)` source channels use `ops[1]` and the second `length(H)`
channels use `ops[2]`. The two channel groups are combined as a sum when
`isfermion=true` and as a difference otherwise.
"""
function dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap};
                    verbose=true,
                    gf_path::String="./",
                    times::AbstractRange=0:0.05:5.0,
                    n::Integer=3,
                    trscheme=truncerror(;rtol=1e-3),
                    approxalg = myDMRG2(),
                    tdvp1 = myTDVP,
                    tdvp2 = myTDVP2(trscheme),
                    isfermion::Bool = true
                    )
    !isdir(gf_path) && mkdir(gf_path)
    gsenergy = expectation_value(gs, H)
    gf = SharedArray{ComplexF64, 3}(length(H), 2*length(H), length(times))
    @sync @distributed for j in 1:2*length(H)
        cur_op = j <= length(H) ? ops[1] : ops[2]
        idx = j <= length(H) ? j : j - length(H)
        ket = chargedMPS(cur_op, gs, idx, approxalg)
        start_time, record_start = now(), now()
        sd = sweep_dot(gs, cur_op, ket)
        gf[:,j,1] = j <= length(H) ? -im * exp(im*gsenergy*times[1]) .* sd : -im * exp(-im*gsenergy*times[1]) .* conj.(sd)
        flush(stdout)
        filename = joinpath(gf_path, "gf_tmax=$(times[end])_id=$(j).jld2")
        if isfile(filename)
            gfb = load(filename)
            for k in 2:length(times)
                if "pro_$(k)" in collect(keys(gfb))
                    gf[:,j,k] = gfb["pro_$(k)"]
                else
                    @warn "Key 'pro_$(k)' not found in $(filename)"
                end
            end
            verbose && println("gf_tmax=$(times[end])_id=$(j).jld2 has loaded!")
            flush(stdout)
            continue
        else
            jldopen(filename, "w") do f
                f["pro_1"] = gf[:,j,1]
            end
        end
        verbose && println("[1/$(length(times))] Started: time evolves 0 of ket$(j) ", Dates.format(start_time, "d.u yyyy HH:MM"))
        envs = environments(ket, H)
        for k in 2:length(times)
            alg = k > n ? tdvp1 : tdvp2
            ket, envs = timestep(ket, H, 0, times[k]-times[k-1], alg, envs)
            sd = sweep_dot(gs, cur_op, ket)
            gf[:,j,k] = j <= length(H) ? -im * exp(im*gsenergy*times[k]) .* sd : -im * exp(-im*gsenergy*times[k]) .* conj.(sd)
            current_time = now()
            verbose && println("[$(k)/$(length(times))] time evolves $(times[k]) of ket$(j) ", " | duration:", Dates.canonicalize(current_time-start_time))
            flush(stdout)
            jldopen(filename, "a") do f
                f["pro_$(k)"] = gf[:,j,k]
            end
            start_time = current_time
        end
        ket = nothing
        envs = nothing
        GC.gc()
        verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    end
    gfs = isfermion ? (gf[:,1:length(H),:] .+ gf[:,(length(H)+1):2*length(H),:]) : (gf[:,1:length(H),:] .- gf[:,(length(H)+1):2*length(H),:])
    return gfs
end

"""
    dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian,
                op::AbstractTensorMap, id::Integer; kwargs...)

Compute a finite-temperature dynamical correlation from one source channel.

The purification `rho` and the charged source ket are evolved together through
`times`. The result has size `(length(H), length(times))` and is multiplied by
`-im` before returning. The `rho_path` keyword is accepted for compatibility
but this implementation does not cache the full `rho(t)` trajectory.
"""
function dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian, op::AbstractTensorMap, id::Integer;
                    verbose=true,
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    rho_path::Union{Nothing, String}=nothing,
                    n::Integer=3, 
                    trscheme=truncerror(;rtol=1e-3),
                    tdvp1 = myTDVP,
                    tdvp2 = myTDVP2(trscheme),
                    )
    !isdir(gf_path) && mkpath(gf_path)
    L, nt = length(H), length(times)
    idx = _dcorrelator_site(id, L)
    gf = zeros(ComplexF64, L, nt)
    Z = dot(rho, rho)
    filename = joinpath(gf_path, "gf_β=$(beta)_tmax=$(times[end])_id=$(id).jld2")

    _dcorrelator_load_complete!(gf, filename, nt; verbose=verbose) && return -im * gf

    ket = chargedMPS(op, rho, idx)
    ket_env = environments(ket, H)
    rho_t = rho
    rho_env = environments(rho_t, H)
    wall_start = now()

    jldopen(filename, "w") do f
        f["times"] = times
        f["id"] = id
        f["beta"] = beta
    end

    for k in 1:nt
        step_start = now()
        for i in 1:L
            bra = chargedMPS(op, rho_t, i)
            gf[i, k] = id <= L ? dot(bra, ket) / Z : dot(ket, bra) / Z
        end

        jldopen(filename, "a") do f
            f["pro_$(k)"] = gf[:, k]
        end

        current_time = now()
        verbose && println("[$k/$nt] finite-T correlation t=$(times[k]) of ket$(id) ",
            " | duration:", Dates.canonicalize(current_time - step_start))
        verbose && flush(stdout)

        if k < nt
            alg = (k + 1) > n ? tdvp1 : tdvp2
            dt = times[k + 1] - times[k]
            rho_t, rho_env = timestep(rho_t, H, 0, dt, alg, rho_env)
            ket, ket_env = timestep(ket, H, 0, dt, alg, ket_env)
        end
    end

    rho_t = nothing
    rho_env = nothing
    ket = nothing
    ket_env = nothing
    GC.gc()
    verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now() - wall_start))
    return -im * gf
end

"""
    dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian,
                op::AbstractTensorMap, indices::AbstractArray; kwargs...)

Compute finite-temperature dynamical correlations for several source channels.

This method keeps only the current thermal state `rho_t`, the active source
kets, and their environments in memory. Completed per-source JLD2 files are
loaded and skipped; incomplete files are recomputed.
"""
function dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true, 
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    rho_path::Union{Nothing, String}=nothing,
                    n::Integer=3, 
                    trscheme=truncerror(;rtol=1e-3),
                    tdvp1 = myTDVP,
                    tdvp2 = myTDVP2(trscheme),
                    )
    !isdir(gf_path) && mkpath(gf_path)
    L, nt = length(H), length(times)
    ids = collect(indices)
    gf = zeros(ComplexF64, L, length(ids), nt)
    Z = dot(rho, rho)

    filenames = Vector{String}(undef, length(ids))
    active = Int[]
    for d in eachindex(ids)
        id = ids[d]
        filename = joinpath(gf_path, "gf_β=$(beta)_tmax=$(times[end])_id=$(id).jld2")
        filenames[d] = filename
        if isfile(filename)
            gfb = load(filename)
            iscomplete = all("pro_$(k)" in keys(gfb) for k in 1:nt)
            if iscomplete
                for k in 1:nt
                    gf[:, d, k] = gfb["pro_$(k)"]
                end
                verbose && println("gf_β=$(beta)_tmax=$(times[end])_id=$(id).jld2 has existed!")
                flush(stdout)
                continue
            end
            @warn "$(filename) is incomplete; recomputing it"
        end
        push!(active, d)
    end

    isempty(active) && return -im * gf

    kets = Vector{FiniteSuperMPS}(undef, length(active))
    ket_envs = Vector{Any}(undef, length(active))
    for a in eachindex(active)
        d = active[a]
        id = ids[d]
        idx = id <= L ? id : id - L
        kets[a] = chargedMPS(op, rho, idx)
        ket_envs[a] = environments(kets[a], H)
        jldopen(filenames[d], "w") do f
            f["times"] = times
            f["id"] = id
            f["beta"] = beta
        end
    end

    rho_t = rho
    rho_env = environments(rho_t, H)
    wall_start = now()
    for k in 1:nt
        step_start = now()
        for i in 1:L
            bra = chargedMPS(op, rho_t, i)
            for a in eachindex(active)
                d = active[a]
                id = ids[d]
                gf[i, d, k] = id <= L ? dot(bra, kets[a]) / Z : dot(kets[a], bra) / Z
            end
        end

        for d in active
            jldopen(filenames[d], "a") do f
                f["pro_$(k)"] = gf[:, d, k]
            end
        end

        current_time = now()
        verbose && println("[$k/$nt] finite-T correlation t=$(times[k]) ",
            " | active sources: $(length(active))",
            " | duration:", Dates.canonicalize(current_time - step_start))
        flush(stdout)

        if k < nt
            alg = (k + 1) > n ? tdvp1 : tdvp2
            dt = times[k + 1] - times[k]
            rho_t, rho_env = timestep(rho_t, H, 0, dt, alg, rho_env)
            for a in eachindex(kets)
                kets[a], ket_envs[a] = timestep(kets[a], H, 0, dt, alg, ket_envs[a])
            end
        end
    end

    rho_t = nothing
    rho_env = nothing
    kets = nothing
    ket_envs = nothing
    GC.gc()
    verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now() - wall_start))
    return -im * gf
end

"""
    dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian,
                ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap};
                    verbose=true, 
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    n::Integer=3, 
                    trscheme=truncerror(;rtol=1e-3),
                    tdvp1 = myTDVP,
                    tdvp2 = myTDVP2(trscheme)
                    )

Compute finite-temperature dynamical correlations for a pair of operators.

The first and second operator channels are evolved as active kets and combined
as `left + right` for fermionic correlations or `left - right` otherwise.
"""
function dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian, ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap};
                    verbose=true, 
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    rho_path::Union{Nothing, String}=nothing,
                    n::Integer=3, 
                    trscheme=truncerror(;rtol=1e-3),
                    tdvp1 = myTDVP,
                    tdvp2 = myTDVP2(trscheme),
                    isfermion::Bool = true
                    )
    !isdir(gf_path) && mkpath(gf_path)
    L, nt = length(H), length(times)
    gf = zeros(ComplexF64, L, 2L, nt)
    Z = dot(rho, rho)

    filenames = Vector{String}(undef, 2L)
    active = Int[]
    for j in 1:2L
        filename = joinpath(gf_path, "gf_β=$(beta)_tmax=$(times[end])_id=$(j).jld2")
        filenames[j] = filename
        if isfile(filename)
            gfb = load(filename)
            iscomplete = all("pro_$(k)" in keys(gfb) for k in 1:nt)
            if iscomplete
                for k in 1:nt
                    gf[:, j, k] = gfb["pro_$(k)"]
                end
                verbose && println("gf_β=$(beta)_tmax=$(times[end])_id=$(j).jld2 has loaded!")
                flush(stdout)
                continue
            end
            @warn "$(filename) is incomplete; recomputing it"
        end
        push!(active, j)
    end

    isempty(active) && begin
        gfs = isfermion ? (gf[:, 1:L, :] .+ gf[:, (L + 1):2L, :]) :
                          (gf[:, 1:L, :] .- gf[:, (L + 1):2L, :])
        return -im * gfs
    end

    kets = Vector{FiniteSuperMPS}(undef, length(active))
    ket_envs = Vector{Any}(undef, length(active))
    for a in eachindex(active)
        j = active[a]
        cur_op = j <= L ? ops[1] : ops[2]
        idx = j <= L ? j : j - L
        kets[a] = chargedMPS(cur_op, rho, idx)
        ket_envs[a] = environments(kets[a], H)
        jldopen(filenames[j], "w") do f
            f["times"] = times
            f["id"] = j
            f["beta"] = beta
        end
    end

    rho_t = rho
    rho_env = environments(rho_t, H)
    wall_start = now()
    for k in 1:nt
        step_start = now()
        has_left = any(active[a] <= L for a in eachindex(active))
        has_right = any(active[a] > L for a in eachindex(active))
        for i in 1:L
            if has_left
                bra₁ = chargedMPS(ops[1], rho_t, i)
                for a in eachindex(active)
                    j = active[a]
                    if j <= L
                        gf[i, j, k] = dot(bra₁, kets[a]) / Z
                    end
                end
            end
            if has_right
                bra₂ = chargedMPS(ops[2], rho_t, i)
                for a in eachindex(active)
                    j = active[a]
                    if j > L
                        gf[i, j, k] = dot(kets[a], bra₂) / Z
                    end
                end
            end
        end

        for j in active
            jldopen(filenames[j], "a") do f
                f["pro_$(k)"] = gf[:, j, k]
            end
        end

        current_time = now()
        verbose && println("[$k/$nt] finite-T correlation t=$(times[k]) ",
            " | active sources: $(length(active))",
            " | duration:", Dates.canonicalize(current_time - step_start))
        flush(stdout)

        if k < nt
            alg = (k + 1) > n ? tdvp1 : tdvp2
            dt = times[k + 1] - times[k]
            rho_t, rho_env = timestep(rho_t, H, 0, dt, alg, rho_env)
            for a in eachindex(kets)
                kets[a], ket_envs[a] = timestep(kets[a], H, 0, dt, alg, ket_envs[a])
            end
        end
    end

    rho_t = nothing
    rho_env = nothing
    kets = nothing
    ket_envs = nothing
    GC.gc()
    verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now() - wall_start))

    gfs = isfermion ? (gf[:, 1:L, :] .+ gf[:, (L + 1):2L, :]) :
                      (gf[:, 1:L, :] .- gf[:, (L + 1):2L, :])
    return -im * gfs
end

"""
    sweep_dot(gs::FiniteNormalMPS, op::AbstractTensorMap, ket::FiniteNormalMPS) -> Vector

Compute `⟨gs|chargedMPO(op,i)†|ket⟩` for all sites `i=1..L` in a single sweep.

# Arguments
- `gs`: the ground state MPS.
- `op`: the local operator. Allowed leg structures:
  - `(1,2)`: `op[-1; -2 -3]` with MPO virtual leg on the right (side=:L convention).
  - `(1,1)`: `op[-1; -2]` diagonal operator without virtual leg.
  Operators with `(2,1)` legs (virtual leg on the left) are not supported.
- `ket`: the time-evolved charged MPS, e.g. `ket = chargedMPS(op, gs, j, approxalg)`.
"""
function sweep_dot(gs::FiniteNormalMPS, op::AbstractTensorMap, ket::FiniteNormalMPS)
    L = length(gs)
    S = promote_type(scalartype(gs), scalartype(ket))
    ncodom = length(codomain(op))
    ndom = length(domain(op))

    if ncodom == 2 && ndom == 1
        throw(ArgumentError("operator with virtual leg on the codomain is not supported; use side=:L convention"))
    end

    if ncodom == 1 && ndom == 2
        # Non-trivial virtual leg: chargedMPO = [I..I | op | Z..Z]
        # gr is MPSTensor (2 codomain, 1 domain), gl is MPSBondTensor (1,1)
        Z = fZ(op)
        gr = Vector{MPSTensor}(undef, L + 1)
        gr[L + 1] = isomorphism(S, right_virtualspace(gs, L) ⊗ domain(Z, 2) ← right_virtualspace(ket, L))
        for j in L:-1:1
            gr[j] = transfer_right(gr[j + 1], Z, gs.AR[j], ket.AR[j])
        end

        gl = Vector{MPSBondTensor}(undef, L + 1)
        gl[1] = isomorphism(S, left_virtualspace(ket, 1) ← left_virtualspace(gs, 1))
        for j in 1:L
            gl[j + 1] = transfer_left(gl[j], gs.AL[j], ket.AL[j])
        end

        G = zeros(S, L)
        for i in 1:L
            @plansor Gi = gl[i][1; 2] * gs.AC[i][2 3; 4] * gr[i+1][4 6; 7] * op[5; 3 6] * conj(ket.AC[i][1 5; 7])
            G[i] = conj(Gi)
        end

    elseif ncodom == 1 && ndom == 1
        # Diagonal operator: chargedMPO = [I..I | op | I..I], no Z string
        # Both gl and gr are MPSBondTensor (1,1)
        gr = Vector{MPSBondTensor}(undef, L + 1)
        gr[L + 1] = isomorphism(S, right_virtualspace(gs, L) ← right_virtualspace(ket, L))
        for j in L:-1:1
            gr[j] = transfer_right(gr[j + 1], gs.AR[j], ket.AR[j])
        end

        gl = Vector{MPSBondTensor}(undef, L + 1)
        gl[1] = isomorphism(S, left_virtualspace(ket, 1) ← left_virtualspace(gs, 1))
        for j in 1:L
            gl[j + 1] = transfer_left(gl[j], gs.AL[j], ket.AL[j])
        end

        G = zeros(S, L)
        for i in 1:L
            @plansor Gi = gl[i][1; 2] * gs.AC[i][2 3; 4] * gr[i+1][4; 7] * op[5; 3] * conj(ket.AC[i][1 5; 7])
            G[i] = conj(Gi)
        end

    else
        throw(ArgumentError("unsupported operator structure ($(ncodom),$(ndom))"))
    end

    return G
end
