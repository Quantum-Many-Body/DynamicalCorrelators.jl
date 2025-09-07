"""
    evolve_mps(H::MPOHamiltonian, ts::AbstractVector, rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H)); 
                        filename::String="default_expiHt_ψ.jld2", 
                    save_id::AbstractArray=[length(ts),], 
                    verbose::Bool=true, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
"""
function evolve_mps(H::MPOHamiltonian, ts::AbstractVector, rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H)); 
                    filename::String="default_expiHt_ψ.jld2", 
                    save_id::AbstractArray=[length(ts),], 
                    verbose::Bool=true, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
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
    dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true, 
                    gf_path::String="./", 
                    times::AbstractRange=0:0.05:5.0, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
    Dynamical correlations in zero temperature
"""
function dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true, 
                    gf_path::String="./", 
                    times::AbstractRange=0:0.05:5.0, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
    !isdir(gf_path)&& mkdir(gf_path)
    gsenergy = expectation_value(gs, H)
    gf = SharedArray{ComplexF64, 3}(length(indices), length(H), length(times))
    @sync @distributed for id in indices
        start_time, record_start = now(), now()
        idx = id <= length(H) ? id : (id - length(H))
        ket = chargedMPS(op, gs, idx)
        gf[id,:,1] = id <= length(H) ? [dot(chargedMPS(op, gs, i), ket) for i in 1:length(H)] : [conj(dot(chargedMPS(op, gs, i), ket)) for i in 1:length(H)]
        filename = joinpath(gf_path, "gf_tmax=$(times[end])_id=$(id).jld2")
        jldopen(filename, "w") do f
        f["pro_1"] = gf[id,:,1]
        end
        verbose && println("[1/$(length(times))] Started: time evolves 0 of ket$(id) ", Dates.format(start_time, "d.u yyyy HH:MM"))
        flush(stdout)
        envs = environments(ket, H)
        for i in 2:length(times)
            alg = i > n ? tdvp1 : tdvp2
            ket, envs = timestep(ket, H, 0, times[i]-times[i-1], alg, envs)
            for j in 1:length(H)
                gf[id,j,i] = (id <= length(H)) ? -im*exp(im*gsenergy*times[i])*dot(chargedMPS(op, gs, j), ket) : -im*exp(-im*gsenergy*times[i])*conj(dot(chargedMPS(op, gs, j), ket))
            end
            current_time = now()
            verbose && println("[$(i)/$(length(times))] time evolves $(times[i]) of ket$(id) ", " | duration:", Dates.canonicalize(current_time-start_time))
            flush(stdout)
            jldopen(filename, "a") do f
                f["pro_$(i)"] = gf[id,:,i]
            end
            start_time = current_time
        end
        ket = nothing
        envs = nothing
        tensorfree!(ket)
        tensorfree!(envs)
        GC.gc()
        verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    end
    gfs = zeros(ComplexF64, length(indices), length(H), length(times))
    gfs .= gf
    return gf
end

"""
    dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true, 
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
    Dynamical correlations in finite temperature
"""
function dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true, 
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    rho_path::String="./beta=$(beta)",
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme),
                    )
    !isdir(gf_path)&& mkdir(gf_path)
    gf = SharedArray{ComplexF64, 3}(length(indices), length(H), length(times))
    Z = dot(rho, rho)
    !isdir(rho_path)&& mkdir(rho_path)
    rho_filename = joinpath(rho_path, "rho_β=$(beta)_tmax=$(times[end]).jld2")
    if isfile(rho_filename)
        rhos = load(rho_filename, "rhos")
        verbose && println("rhos is successfully loaded")
        flush(stdout)
    else
        rhos = Vector{FiniteSuperMPS}(undef, length(times))
        rhos[1] = rho
        env = environments(rho, H)
        for i in 2:length(times)
            alg = i > n ? tdvp1 : tdvp2
            rho, env = timestep(rho, H, 0, times[i]-times[i-1], alg, env)
            verbose && println("[$(i-1)/$(length(times)-1)] evolves t=$(times[i]) of rho ", Dates.format(now(), "d.u yyyy HH:MM"))
            rhos[i] = rho
            flush(stdout)
        end
        save(rho_filename, "rhos", rhos)
    end
    env = nothing
    tensorfree!(env)
    @sync @distributed for id in indices
        start_time, record_start = now(), now()
        idx = id <= length(H) ? id : (id - length(H))
        ket = chargedMPS(op, rhos[1], idx)
        gf[id,:,1] = (id <= length(H)) ? [dot(chargedMPS(op, rhos[1], i), ket)/Z for i in 1:length(H)] : [conj(dot(chargedMPS(op, rhos[1], i), ket))/Z for i in 1:length(H)]
        flush(stdout)
        filename = joinpath(gf_path, "gf_β=$(beta)_tmax=$(times[end])_id=$(id).jld2")
        jldopen(filename, "w") do f
        f["pro_1"] = gf[id,:,1]
        end
        verbose && println("[1/$(length(times))] Started: time evolves 0 of ket$(id) ", Dates.format(start_time, "d.u yyyy HH:MM"))
        envs = environments(ket, H)
        for i in 2:length(times)
            alg = i > n ? tdvp1 : tdvp2
            ket, envs = timestep(ket, H, 0, times[i]-times[i-1], alg, envs)
            for j in 1:length(H)
                gf[id,j,i] = (id <= length(H)) ? dot(chargedMPS(op, rhos[i], j), ket)/Z : conj(dot(chargedMPS(op, rhos[i], j), ket))/Z
            end
            current_time = now()
            verbose && println("[$(i)/$(length(times))] time evolves $(times[i]) of ket$(id) ", " | duration:", Dates.canonicalize(current_time-start_time))
            flush(stdout)
            jldopen(filename, "a") do f
                f["pro_$(i)"] = gf[id,:,i]
            end
            start_time = current_time
        end
        ket = nothing
        envs = nothing
        tensorfree!(ket)
        tensorfree!(envs)
        GC.gc()
        verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    end
    gfs = zeros(ComplexF64, length(indices), length(H), length(times))
    gfs .= gf
    return gfs
end

"""
    dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, mps::AbstractVector{<:FiniteNormalMPS};
                    verbose=true, 
                    gf_path::String="./", 
                    times::AbstractRange=0:0.05:5.0, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
    Dynamical correlations in zero temperature
"""
function dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, mps::AbstractVector{<:FiniteNormalMPS};
                    verbose=true, 
                    gf_path::String="./", 
                    times::AbstractRange=0:0.05:5.0, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
    !isdir(gf_path)&& mkdir(gf_path)
    gsenergy = expectation_value(gs, H)
    gf = SharedArray{ComplexF64, 3}(length(mps), length(mps)÷2, length(times))
    @sync @distributed for id in 1:length(mps)
        start_time, record_start = now(), now()
        ket = mps[id]
        gf[id,:,1] = id <= length(H) ? [dot(bra, ket) for bra in mps[1:length(H)]] : [conj(dot(bra, ket)) for bra in mps[(length(H)+1):end]]
        flush(stdout)
        filename = joinpath(gf_path, "gf_tmax=$(times[end])_id=$(id).jld2")
        jldopen(filename, "w") do f
        f["pro_1"] = gf[id,:,1]
        end
        verbose && println("[1/$(length(times))] Started: time evolves 0 of ket$(id) ", Dates.format(start_time, "d.u yyyy HH:MM"))
        envs = environments(ket, H)
        for i in 2:length(times)
            alg = i > n ? tdvp1 : tdvp2
            ket, envs = timestep(ket, H, 0, times[i]-times[i-1], alg, envs)
            for j in 1:(length(mps)÷2)
                gf[id,j,i] = (id <= length(H)) ? -im*exp(im*gsenergy*times[i])*dot(mps[1:length(H)][j], ket) : -im*exp(-im*gsenergy*times[i])*conj(dot(mps[(length(H)+1):end][j], ket))
            end
            current_time = now()
            verbose && println("[$(i)/$(length(times))] time evolves $(times[i]) of ket$(id) ", " | duration:", Dates.canonicalize(current_time-start_time))
            flush(stdout)
            jldopen(filename, "a") do f
                f["pro_$(i)"] = gf[id,:,i]
            end
            start_time = current_time
        end
        ket = nothing
        envs = nothing
        tensorfree!(ket)
        tensorfree!(envs)
        GC.gc()
        verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    end
    gfs = zeros(ComplexF64, length(mps), length(mps)÷2, length(times))
    gfs .= gf
    return gf
end

"""
    dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true, 
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
    Dynamical correlations in finite temperature
"""
function dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian, ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap};
                    verbose=true, 
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    rho_path::String="./beta=$(beta)",
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
    !isdir(gf_path)&& mkdir(gf_path)
    gf = SharedArray{ComplexF64, 3}(2*length(H), length(H), length(times))
    Z = dot(rho, rho)
    !isdir(rho_path)&& mkdir(rho_path)
    rho_filename = joinpath(rho_path, "rho_β=$(beta)_tmax=$(times[end]).jld2")
    if isfile(rho_filename)
        rhos = load(rho_filename, "rhos")
        verbose && println("rhos is successfully loaded")
        flush(stdout)
    else
        rhos = Vector{FiniteSuperMPS}(undef, length(times))
        rhos[1] = rho
        env = environments(rho, H)
        for i in 2:length(times)
            alg = i > n ? tdvp1 : tdvp2
            rho, env = timestep(rho, H, 0, times[i]-times[i-1], alg, env)
            verbose && println("[$(i-1)/$(length(times)-1)] evolves t=$(times[i]) of rho ")
            rhos[i] = rho
            flush(stdout)
        end
        save(rho_filename, "rhos", rhos)
    end
    env = nothing
    tensorfree!(env)
    @sync @distributed for id in 1:2*length(H)
        start_time, record_start = now(), now()
        ket = id <= length(H) ? chargedMPS(ops[1], rhos[1], id) : chargedMPS(ops[2], rhos[1], id-length(H))
        gf[id,:,1] = (id <= length(H)) ? [dot(chargedMPS(ops[1], rhos[1], i), ket)/Z for i in 1:length(H)] : [conj(dot(chargedMPS(ops[2], rhos[1], i), ket))/Z for i in 1:length(H)]
        flush(stdout)
        filename = joinpath(gf_path, "gf_β=$(beta)_tmax=$(times[end])_id=$(id).jld2")
        jldopen(filename, "w") do f
            f["pro_1"] = gf[id,:,1]
        end
        verbose && println("[1/$(length(times))] Started: time evolves 0 of ket$(id) ", Dates.format(start_time, "d.u yyyy HH:MM"))
        envs = environments(ket, H)
        for i in 2:length(times)
            alg = i > n ? tdvp1 : tdvp2
            ket, envs = timestep(ket, H, 0, times[i]-times[i-1], alg, envs)
            for j in 1:length(H)
                gf[id,j,i] = (id <= length(H)) ? dot(chargedMPS(ops[1], rhos[i], j), ket)/Z : conj(dot(chargedMPS(ops[2], rhos[i], j), ket))/Z
            end
            current_time = now()
            verbose && println("[$(i)/$(length(times))] time evolves $(times[i]) of ket$(id) ", " | duration:", Dates.canonicalize(current_time-start_time))
            flush(stdout)
            jldopen(filename, "a") do f
                f["pro_$(i)"] = gf[id,:,i]
            end
            start_time = current_time
        end
        ket = nothing
        envs = nothing
        tensorfree!(ket)
        tensorfree!(envs)
        GC.gc()
        verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    end
    gfs = zeros(ComplexF64, 2*length(H), length(H), length(times))
    gfs .= gf
    return gfs
end
