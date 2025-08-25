"""
    expiHt(H::MPOHamiltonian, ts::AbstractVector, rho::FiniteMPO=identityMPO(H); filename::String="default_expHt.jld2", save_all::Bool=false, verbose::Bool=true, n::Integer=3, trscheme=truncerr(1e-3))
"""
function expiHt(H::MPOHamiltonian, ts::AbstractVector, rho::FiniteMPO=identityMPO(H); filename::String="default_expHt.jld2", save_all::Bool=false, verbose::Bool=true, n::Integer=3, trscheme=truncerr(1e-3))
    rho_mps = convert(FiniteMPS, rho)
    start_time, record_start = now(), now()
    verbose && println("[1/$(length(ts))] t = $(ts[1]) ", " | Started:", Dates.format(start_time, "d.u yyyy HH:MM"))
    flush(stdout)
    envs = environments(rho_mps, H)
    jldopen(filename, "w") do f
        f["t=$(ts[1])"] = rho
    end
    for i in 2:length(ts)
        alg = i > n ? DefaultTDVP : DefaultTDVP2(trscheme)
        rho_mps, envs = timestep(rho_mps, H, 0, ts[i]-ts[i-1], alg, envs)
        current_time = now()
        verbose && println("[$i/$(length(ts))] t = $(ts[i]) ", " | duration:", Dates.canonicalize(current_time-start_time))
        flush(stdout)
        jldopen(filename, "a") do f
            if save_all || i == length(ts)
                f["t=$(ts[i])"] = convert(FiniteMPO, rho_mps)
            end
        end
        start_time = current_time
    end
    verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    return convert(FiniteMPO, rho_mps)
end


function A_expiHt_B(gs::AbstractFiniteMPS, H::MPOHamiltonian, expiht_data, op, indices::AbstractArray;
                    verbose=true, 
                    rev::Bool=false,
                    times::AbstractArray,
                    filename::String="default_gf.jld2")
    gsenergy = expectation_value(gs, H)
    gf = SharedArray{ComplexF64, 3}(length(indices), length(H), length(times))
    start_time, record_start = now(), now()
    verbose && println("Started:", Dates.format(record_start, "d.u yyyy HH:MM"))
    flush(stdout)
    jldopen(filename, "w") do f
        f["times"] = times
    end
    for i in eachindex(times)
        rho = expiht_data["t=$(times[i])"]
        indices2 = CartesianIndices((length(indices), length(H)))
        @sync @distributed for idx in 1:length(indices2)
            j, k = indices2[idx].I
            if isa(op, Function)
                if (j+k) <= length(H)
                    gf[j,k,i] = dot(chargedMPS(op(:R),gs,k), rho, chargedMPS(op(:R),gs,j))
                else
                    gf[j,k,i] = dot(chargedMPS(op(:L),gs,k), rho, chargedMPS(op(:L),gs,j))
                end
            else
                gf[j,k,i] = dot(chargedMPS(op,gs,k), rho, chargedMPS(op,gs,j))
            end
        end
        GC.gc()
        factor₁, factor₂ = -im*exp(im*gsenergy*times[i]), -im*exp(-im*gsenergy*times[i])
        gf[:,:,i] = rev ? factor₂*conj.(gf[:,:,i]) : factor₁*gf[:,:,i]
        jldopen(filename, "a") do f
            f["gf_$(times[i])"] = gf[:,:,i]
        end
        verbose && println("[$i/$(length(times))] gf_$(times[i]) is done ", " | duration:", Dates.canonicalize(now()  - start_time))
        start_time = now()
    end
    verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    jldopen(filename, "a") do f
        f["gfs_rev=$(rev)"] = gf
    end
    return gf
end


function propagator(H::MPOHamiltonian, bra::FiniteMPS, ket::FiniteMPS; rev::Bool=false, imag::Bool=false,
                    times::AbstractRange=0:0.05:5.0, 
                    dt=round(times.step.hi, digits=2), 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3))
    propagators = zeros(ComplexF64, length(times))
    propagators[1] = dot(bra, ket)
    envs = environments(ket, H)
    if imag
        for (i, t) in enumerate(times[2:end])
            alg = t > n * dt ? DefaultTDVP : DefaultTDVP2(trscheme)
            ket, envs = timestep(ket, H, 0, -1im*dt, alg, envs)
            propagators[i+1] = dot(bra, ket)
        end
    else
        for (i, t) in enumerate(times[2:end])
            alg = t > n * dt ? DefaultTDVP : DefaultTDVP2(trscheme)
            ket, envs = timestep(ket, H, 0, dt, alg, envs)
            propagators[i+1] = dot(bra, ket)
        end
    end
    rev ? propagators = conj.(propagators) : propagators = propagators
    return propagators
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

