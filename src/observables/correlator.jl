"""
    propagator(H::MPOHamiltonian, bra::FiniteMPS, ket::FiniteMPS; rev::Bool=false, imag::Bool=false, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncerr(1e-3))
    propagator(H::MPOHamiltonian, bras::Vector{<:FiniteMPS}, ket::FiniteMPS; verbose::Bool=false, rev::Bool=false, imag::Bool=false, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncerr(1e-3))
"""
function propagator(H::MPOHamiltonian, bra::FiniteMPS, ket::FiniteMPS; rev::Bool=false, imag::Bool=false, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncerr(1e-3))
    times = collect(0:dt:ft)
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

function propagator(H::MPOHamiltonian, bras::Vector{<:FiniteMPS}, ket::FiniteMPS; id::Union{String, Integer}="*", savekets::Bool=false, filename::String="default_gf_slice.jld2", verbose::Bool=false, rev::Bool=false, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncerr(1e-3))
    times = collect(0:dt:ft)
    propagators = zeros(ComplexF64, length(bras), length(times))
    propagators[:,1] = [dot(bras[i], ket) for i in 1:length(bras)]
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
        for j in eachindex(bras)
            propagators[j,i+1] = dot(bras[j], ket)
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

struct RetardedGF{K} end
RetardedGF(::Type{RetardedGF{:f}}) = 1
RetardedGF(::Type{RetardedGF{:b}}) = -1


function dcorrelator(::Type{R}, H::MPOHamiltonian, 
                    gsenergy::Number, mps::Vector{<:FiniteMPS};
                    verbose=true, 
                    path::String="./", 
                    savekets=false,  
                    parallel::Union{String, Integer}=Threads.nthreads(), 
                    dt::Number=0.05, 
                    ft::Number=5.0, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3)) where R<:RetardedGF
    t, half = collect(0:dt:ft), length(mps)÷2
    if parallel == 1
        gf = zeros(ComplexF64, length(mps), half, length(0:dt:ft))
        for i in 1:length(mps)
            if i <= half
                gf[i,:,:] = propagator(H, mps[1:half], mps[i]; id=i, filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
            else
                gf[i,:,:] = propagator(H, mps[(half+1):end], mps[i]; id=i, filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme)
            end
        end
    else
        gf = zeros(ComplexF64, length(mps), half, length(0:dt:ft))
        idx = Threads.Atomic{Int}(1)
        n = length(mps)
        Threads.@sync for _ in 1:parallel
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1) 
                i > n && break  
                if i <= half
                    gf[i,:,:] = propagator(H, mps[1:half], mps[i]; id=i, filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
                else
                    gf[i,:,:] = propagator(H, mps[(half+1):end], mps[i]; id=i, filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme)
                end
            end
        end
    end
    for i in eachindex(t)
        factor₁, factor₂ = -im*exp(im*gsenergy*t[i]), -im*exp(-im*gsenergy*t[i])
        gf[1:half,:,i] = factor₁*gf[1:half,:,i]
        gf[(half+1):end,:,i] = factor₂*gf[(half+1):end,:,i]
    end
    return gf[1:half,:,:] + RetardedGF(R)*gf[(half+1):end,:,:]
end

function dcorrelator(::Type{R}, gf_slices::AbstractArray{<:AbstractMatrix}, gsenergy::Number, t::AbstractRange) where R<:RetardedGF
    half = length(gf_slices)÷2
    gf = zeros(ComplexF64, length(gf_slices), half, length(t))
    for i in eachindex(gf_slices)
        gf[i,:,:] .= gf_slices[i]
    end
    for i in eachindex(t)
        factor₁, factor₂ = -im*exp(im*gsenergy*t[i]), -im*exp(-im*gsenergy*t[i])
        gf[1:half,:,i] = factor₁*gf[1:half,:,i]
        gf[(half+1):end,:,i] = factor₂*gf[(half+1):end,:,i]
    end
    return gf[1:half,:,:] + RetardedGF(R)*gf[(half+1):end,:,:]
end

struct GreaterLessGF end

function dcorrelator(::Type{GreaterLessGF}, H::MPOHamiltonian, gsenergy::Number, mps::Vector{<:FiniteMPS}; whichs=:both, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncerr(1e-3))
    t, half = collect(0:dt:ft), length(mps)÷2
    if whichs == :both
        gf = SharedArray{ComplexF64, 3}(length(mps), half, length(0:dt:ft))
        @sync @distributed for i in 1:length(mps)
            if i <= half
                gf[i,:,:] = propagator(H, mps[1:half], mps[i]; rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
            else
                gf[i,:,:] =  propagator(H, mps[(half+1):end], mps[i]; rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme)
            end
        end
        for i in eachindex(t)
            factor₁, factor₂ = -im*exp(im*gsenergy*t[i]), -im*exp(-im*gsenergy*t[i])
            gf[1:half,:,i] = factor₁*gf[1:half,:,i]
            gf[(half+1):end,:,i] = factor₂*gf[(half+1):end,:,i]
        end
        return [gf[1:half,:,:], gf[(half+1):end,:,:]]
    elseif whichs == :greater
        gf = SharedArray{ComplexF64, 3}(length(mps), length(mps), length(0:dt:ft))
        @sync @distributed for i in 1:length(mps)
            gf[i,:,:] = propagator(H, mps[1:end], mps[i]; rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
        end
        for i in eachindex(t)
            gf[:,:,i] = -im*exp(im*gsenergy*t[i])*gf[:,:,i]
        end
        return gf
    elseif whichs == :less
        gf = SharedArray{ComplexF64, 3}(length(mps), length(mps), length(0:dt:ft))
        @sync @distributed for i in 1:length(mps)
            gf[i,:,:] =  propagator(H, mps[1:end], mps[i]; rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme)
        end
        for i in eachindex(t)
            gf[:,:,i] = -im*exp(-im*gsenergy*t[i])*gf[:,:,i]
        end
        return gf
    end
end



struct GorkovGF end

struct MatsubaraGF end

function dcorrelator(::Type{MatsubaraGF}, H::MPOHamiltonian, gsenergy::Number, mps::Vector{<:FiniteMPS}; dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncerr(1e-3))
    t = collect(0:dt:ft)
    gf = SharedArray{ComplexF64, 3}(length(mps), length(mps), length(0:dt:ft))
    @sync @distributed for i in 1:length(mps)
        gf[i,:,:] = propagator(H, mps[1:end], mps[i]; rev=false, imag=true, dt=dt, ft=ft, n=n, trscheme=trscheme) 
    end
    for i in eachindex(t)
        gf[:,:,i] = exp(gsenergy*t[i])*gf[:,:,i]
    end
    return gf
end