"""
    evolve_mps(H::MPOHamiltonian, ts::AbstractVector, rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H)); filename::String="default_expiHt_ψ.jld2", save_all::Bool=false, verbose::Bool=true, n::Integer=3, trscheme=truncerr(1e-3))
"""
function evolve_mps(H::MPOHamiltonian, ts::AbstractVector, rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H)); filename::String="default_expiHt_ψ.jld2", save_all::Bool=false, verbose::Bool=true, n::Integer=3, trscheme=truncerr(1e-3))
    start_time, record_start = now(), now()
    verbose && println("[1/$(length(ts))] t = $(ts[1]) ", " | Started:", Dates.format(start_time, "d.u yyyy HH:MM"))
    flush(stdout)
    envs = environments(rho_mps, H)
    jldopen(filename, "w") do f
        f["t=$(ts[1])"] = rho_mps
    end
    for i in 2:length(ts)
        alg = i > n ? DefaultTDVP : DefaultTDVP2(trscheme)
        rho_mps, envs = timestep(rho_mps, H, 0, ts[i]-ts[i-1], alg, envs)
        current_time = now()
        verbose && println("[$i/$(length(ts))] t = $(ts[i]) ", " | duration:", Dates.canonicalize(current_time-start_time))
        flush(stdout)
        jldopen(filename, "a") do f
            if save_all || i == length(ts)
                f["t=$(ts[i])"] = rho_mps
            end
        end
        start_time = current_time
    end
    verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    return rho_mps
end


function dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true, 
                    path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    n::Integer=3, 
                    trscheme=truncerr(1e-3))
    gf = SharedArray{ComplexF64, 3}(length(indices), length(H), length(times))
    Z = dot(rho, rho)
    @sync @distributed for id in indices
        start_time, record_start = now(), now()
        idx = id <= length(H) ? id : (id - length(H))
        ket = chargedMPS(op, rho, idx)
        gf[id,:,1] = [dot(chargedMPS(op, rho, i),chargedMPS(op, rho, idx))/Z for i in 1:length(H)]
        flush(stdout)
        filename = joinpath(path, "gf_β=$(beta)_tmax=$(times[end])_id=$(id).jld2")
        jldopen(filename, "w") do f
        f["pro_1"] = gf[id,:,1]
        end
        verbose && println("[1/$(length(times))] Started: time evolves 0 of ket$(id) ", Dates.format(start_time, "d.u yyyy HH:MM"))
        envs = environments(ket, H)
        envs2 = environments(rho, H)
        for i in 2:length(times)
            alg = i > n ? DefaultTDVP : DefaultTDVP2(trscheme)
            ket, envs = timestep(ket, H, 0, times[i]-times[i-1], alg, envs)
            rho, = timestep(rho, H, 0, times[i]-times[i-1], alg, envs2)
            for j in 1:length(H)
                gf[id,j,i] = (id <= length(H)) ? dot(chargedMPS(op, rho, j), ket)/Z : conj(dot(chargedMPS(op, rho, j), ket))/Z
            end
            current_time = now()
            verbose && println("[$(i)/$(length(times))] time evolves $(times[i]) of ket$(id) ", " | duration:", Dates.canonicalize(current_time-start_time))
            flush(stdout)
            jldopen(filename, "a") do f
                f["pro_$(i)"] = gf[id,:,i]
            end
            start_time = current_time
        end
        verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    end
    gfs = zeros(ComplexF64, length(indices), length(H), length(times))
    gfs .= gf
    return gfs
end

function propagator(gs::AbstractFiniteMPS, H::MPOHamiltonian, op::AbstractTensorMap, id::Integer; 
                savekets::Bool=false, 
                filename::String="default_gf_slice.jld2", 
                verbose::Bool=true, 
                rev::Bool=false, 
                times::AbstractRange=0:0.05:5.0, 
                dt=round(times.step.hi, digits=2), 
                n::Integer=3, 
                trscheme=truncerr(1e-3))
    propagators = zeros(ComplexF64, length(H), length(times))
    idx = id <= length(H) ? id : id - length(H)
    ket = chargedMPS(op, gs, idx)
    propagators[:,1] = [dot(chargedMPS(op, gs, i), ket) for i in 1:length(H)]
    start_time, record_start = now(), now()
    verbose && println("[1/$(length(times))] Started: time evolves 0 of ket$(id) ", Dates.format(start_time, "d.u yyyy HH:MM"))
    flush(stdout)
    envs = environments(ket, H)
    jldopen(filename, "w") do f
        f["pro_1"] = propagators[:, 1]
    end
    for (i, t) in enumerate(times[2:end])
        alg = t > n * dt ? DefaultTDVP : DefaultTDVP2(trscheme)
        ket, envs = timestep(ket, H, 0, dt, alg, envs)
        for j in 1:length(H)
            propagators[j,i+1] = dot(chargedMPS(op, gs, j), ket)
        end
        current_time = now()
        verbose && println("[$(i+1)/$(length(times))] time evolves $(t) of ket$(id) ", " | duration:", Dates.canonicalize(current_time-start_time))
        flush(stdout)
        jldopen(filename, "a") do f
            f["pro_$(i+1)"] = propagators[:, i+1]
            if savekets || t == times[end]
                f["ket_t=$t"] = ket
            end
        end
        start_time = current_time
    end
    record_end = now()
    verbose && println("Ended: ", Dates.format(record_end, "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(record_end-record_start))

    rev ? propagators = conj.(propagators) : propagators = propagators
    return propagators
end

function dcorrelator(gs::AbstractFiniteMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray, rev::Bool;
                    verbose=true, 
                    path::String="./", 
                    savekets=false,  
                    times::AbstractRange=0:0.05:5.0, 
                    dt=round(times.step.hi, digits=2),
                    n::Integer=3, 
                    trscheme=truncerr(1e-3))
    gsenergy = expectation_value(gs, H)
    gf = SharedArray{ComplexF64, 3}(length(indices), length(H), length(times))
    @sync @distributed for i in indices
        gf[i,:,:] = propagator(gs, H, op, i; filename=joinpath(path, "gf_slice_$(i)_$(times[1])_$(dt)_$(times[end]).jld2"), verbose=verbose, savekets=savekets, rev=rev, times=times, dt=dt, n=n, trscheme=trscheme) 
    end
    for i in eachindex(times)
        factor = rev ? -im*exp(-im*gsenergy*times[i]) : -im*exp(im*gsenergy*times[i])
        gf[:,:,i] = factor*gf[:,:,i]
    end
    return gf
end

