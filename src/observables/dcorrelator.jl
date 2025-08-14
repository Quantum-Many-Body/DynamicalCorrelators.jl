"""
    RetardedGF{K}
    RetardedGF(::Type{RetardedGF{:f}}) = 1
    RetardedGF(::Type{RetardedGF{:b}}) = -1
"""
struct RetardedGF{K} end
RetardedGF(::Type{RetardedGF{:f}}) = 1
RetardedGF(::Type{RetardedGF{:b}}) = -1

"""
    GreaterLessGF
"""
struct GreaterLessGF end

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
        f["t=$(ts[i])"] = rho
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
    verbose && println("Ended: ", Dates.format(record_end, "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))

    return convert(FiniteMPO, rho_mps)
end

function A_expiHt_B(::Type{R}, gs::AbstractFiniteMPS, H::MPOHamiltonian, ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap}, ts::AbstractVector, rho::FiniteMPO=identityMPO(H);
                    verbose=true, 
                    filename::String="default_gfs.jld2", 
                    save_all=false,  
                    parallel::Union{String, Integer}=Threads.nthreads(), 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3)) where R<:RetardedGF
    gsenergy = expectation_value(gs, H)
    mps = [[chargedMPS(ops[1], gs, j) for j in 1:length(gs)]; [chargedMPS(ops[2], gs, j) for j in 1:length(gs)]]
    gf = zeros(ComplexF64, (2*length(H)), length(H), length(ts))
    rho_mps = convert(FiniteMPS, rho)
    evolve_start, evolve_finish, record_start = now(), now(), now()
    verbose && println("[1/$(length(ts))] t = $(ts[1]) ", " | Started:", Dates.format(record_start, "d.u yyyy HH:MM"))
    flush(stdout)
    envs = environments(rho_mps, H)
    jldopen(filename, "w") do f
        f["ts"] = ts
    end
    for i in eachindex(ts)
        if i > 1
            alg = i > n ? DefaultTDVP : DefaultTDVP2(trscheme)
            rho_mps, envs = timestep(rho_mps, H, 0, ts[i]-ts[i-1], alg, envs)
            rho = convert(FiniteMPO, rho_mps)
            evolve_finish = now()
            verbose && println("[$i/$(length(ts))] t = $(ts[i]) ", " | duration:", Dates.canonicalize(evolve_finish-evolve_start))
            flush(stdout)
        end

        idx = Threads.Atomic{Int}(1)
        indices = CartesianIndices((2*length(H), length(H)))
        m = length(indices)
        Threads.@sync for _ in 1:parallel
            Threads.@spawn while true
                j = Threads.atomic_add!(idx, 1) 
                j > m && break  
                a, b = indices[j].I
                if a <= length(H)
                    gf[a,b,i] = dot(mps[b], rho, mps[a])
                else
                    gf[a,b,i] = conj(dot(mps[b], rho, mps[a-length(H)]))
                end
            end
        end
        factor₁, factor₂ = -im*exp(im*gsenergy*ts[i]), -im*exp(-im*gsenergy*ts[i])
        gf[1:length(H),:,i] = factor₁*gf[1:length(H),:,i]
        gf[(length(H)+1):end,:,i] = factor₂*gf[(length(H)+1):end,:,i]
        GC.gc()
        jldopen(filename, "a") do f
            f["gf_$(ts[i])"] = gf[:,:,i]
            if save_all || i == length(ts)
                f["rho_$(ts[i])"] = rho
            end
        end
        verbose && println("    gf_$(ts[i]) is done ", " | duration:", Dates.canonicalize(now() - evolve_finish))
        evolve_start = now()
    end

    verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    gfs = gf[1:length(H),:,:] + RetardedGF(R)*gf[(length(H)+1):end,:,:]
    jldopen(filename, "a") do f
        f["gfs"] = gfs
    end
    return gfs
end

function A_expiHt_B(::Type{GreaterLessGF}, gs::AbstractFiniteMPS, H::MPOHamiltonian, ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap}, ts::AbstractVector, rho::FiniteMPO=identityMPO(H);
                    verbose=true, 
                    which=:greater,
                    filename::String="default_gfs.jld2", 
                    save_all=false,  
                    parallel::Union{String, Integer}=Threads.nthreads(), 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3))
    gsenergy = expectation_value(gs, H) 
    gf = zeros(ComplexF64, (length(H)), length(H), length(ts))
    rho_mps = convert(FiniteMPS, rho)
    mps = which==:greater ? [chargedMPS(ops[1], gs, j) for j in 1:length(gs)] : [chargedMPS(ops[2], gs, j) for j in 1:length(gs)]
    evolve_start, evolve_finish, record_start = now(), now(), now()
    verbose && println("[1/$(length(ts))] t = $(ts[1]) ", " | Started:", Dates.format(record_start, "d.u yyyy HH:MM"))
    flush(stdout)
    envs = environments(rho_mps, H)
    jldopen(filename, "w") do f
        f["ts"] = ts
    end
    for i in eachindex(ts)
        if i > 1
            alg = i > n ? DefaultTDVP : DefaultTDVP2(trscheme)
            rho_mps, envs = timestep(rho_mps, H, 0, ts[i]-ts[i-1], alg, envs)
            rho = convert(FiniteMPO, rho_mps)
            evolve_finish = now()
            verbose && println("[$i/$(length(ts))] t = $(ts[i]) ", " | duration:", Dates.canonicalize(evolve_finish-evolve_start))
            flush(stdout)
        end

        if parallel == 1
            for j in axes(gf, 1)
                for k in axes(gf, 2)
                    gf[j,k,i] = which==:greater ? -im*exp(im*gsenergy*ts[i])*dot(mps[k], rho, mps[j]) : -im*exp(-im*gsenergy*ts[i])*conj(dot(mps[k], rho, mps[j]))
                end
            end
            GC.gc()
        else
            idx = Threads.Atomic{Int}(1)
            indices = CartesianIndices((length(H), length(H)))
            m = length(indices)
            Threads.@sync for _ in 1:parallel
                Threads.@spawn while true
                    j = Threads.atomic_add!(idx, 1) 
                    j > m && break  
                    a, b = indices[j].I
                    gf[a,b,i] = which==:greater ? -im*exp(im*gsenergy*ts[i])*dot(mps[b], rho, mps[a]) : -im*exp(-im*gsenergy*ts[i])*conj(dot(mps[b], rho, mps[a]))
                end
            end
            GC.gc()
        end

        jldopen(filename, "a") do f
            f["gf_$(ts[i])"] = gf[:,:,i]
            if save_all || i == length(ts)
                f["rho_$(ts[i])"] = rho
            end
        end
        verbose && println("    gf_$(ts[i]) is done ", " | duration:", Dates.canonicalize(now() - evolve_finish))
        evolve_start = now()
    end

    verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    jldopen(filename, "a") do f
        f["gfs"] = gf
    end
    return gf
end

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

"""
    propagator(gs::AbstractFiniteMPS, H::MPOHamiltonian, op::AbstractTensorMap, id::Integer; savekets::Bool=false, filename::String="default_gf_slice.jld2", verbose::Bool=true, rev::Bool=false, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncerr(1e-3))
"""
function propagator(gs::AbstractFiniteMPS, H::MPOHamiltonian, op::AbstractTensorMap, id::Integer; savekets::Bool=false, filename::String="default_gf_slice.jld2", verbose::Bool=true, rev::Bool=false, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncerr(1e-3))
    times = collect(0:dt:ft)
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

"""
    propagator(H::MPOHamiltonian, bras::Vector{<:FiniteMPS}, ket::FiniteMPS; savekets::Bool=false, filename::String="default_gf_slice.jld2", verbose::Bool=true, rev::Bool=false, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncerr(1e-3))
"""
function propagator(H::MPOHamiltonian, bras::Vector{<:FiniteMPS}, ket::FiniteMPS; savekets::Bool=false, filename::String="default_gf_slice.jld2", verbose::Bool=true, rev::Bool=false, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncerr(1e-3))
    times = collect(0:dt:ft)
    propagators = zeros(ComplexF64, length(bras), length(times))
    propagators[:,1] = [dot(bras[i], ket) for i in 1:length(bras)]
    start_time, record_start = now(), now()
    verbose && println("[1/$(length(times))] Started: time evolves 0 ", Dates.format(start_time, "d.u yyyy HH:MM"))
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
        verbose && println("[$(i+1)/$(length(times))] time evolves $(t)", " | duration:", Dates.canonicalize(current_time-start_time))
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
    jldopen(filename, "a") do f
        f["pros"] = propagators
    end
    return propagators
end



"""
    dcorrelator(::Type{R}, gs::AbstractFiniteMPS, H::MPOHamiltonian, ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap})
"""
function dcorrelator(::Type{R}, gs::AbstractFiniteMPS, H::MPOHamiltonian, ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap};
                    verbose=true, 
                    path::String="./", 
                    savekets=false,  
                    parallel::Union{String, Integer}=Threads.nthreads(), 
                    dt::Number=0.05, 
                    ft::Number=5.0, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3)) where R<:RetardedGF
    t, half = collect(0:dt:ft), length(H)
    gsenergy = expectation_value(gs, H)
    if parallel == "np"
        gf = SharedArray{ComplexF64, 3}(2*half, half, length(0:dt:ft))
        @sync @distributed for i in 1:(2*half)
            if i <= half
                gf[i,:,:] = propagator(gs, H, ops[1], i; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
            else
                gf[i,:,:] = propagator(gs, H, ops[2], i; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme)
            end
        end
    elseif parallel == 1
        gf = zeros(ComplexF64, (2*half), half, length(0:dt:ft))
        for i in 1:(2*half)
            if i <= half
                gf[i,:,:] = propagator(gs, H, ops[1], i; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
            else
                gf[i,:,:] = propagator(gs, H, ops[2], i; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme)
            end
        end
    else
        gf = zeros(ComplexF64, (2*half), half, length(0:dt:ft))
        idx = Threads.Atomic{Int}(1)
        n = (2*half)
        Threads.@sync for _ in 1:parallel
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1) 
                i > n && break  
                if i <= half
                    gf[i,:,:] = propagator(gs, H, ops[1], i; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
                else
                    gf[i,:,:] = propagator(gs, H, ops[2], i; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme)
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

function dcorrelator(gs::AbstractFiniteMPS, H::MPOHamiltonian, op::AbstractTensorMap, rev::Bool;
                    verbose=true, 
                    path::String="./", 
                    savekets=false,  
                    parallel::Union{String, Integer}=Threads.nthreads(), 
                    dt::Number=0.05, 
                    ft::Number=5.0, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3))
    t, half = collect(0:dt:ft), length(H)
    gsenergy = expectation_value(gs, H)
    if parallel == "np"
        gf = SharedArray{ComplexF64, 3}(half, half, length(0:dt:ft))
        @sync @distributed for i in 1:(half)
            gf[i,:,:] = propagator(gs, H, op, i; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=rev, dt=dt, ft=ft, n=n, trscheme=trscheme) 
        end
    elseif parallel == 1
        gf = zeros(ComplexF64, half, half, length(0:dt:ft))
        for i in 1:(2*half)
            gf[i,:,:] = propagator(gs, H, op, i; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=rev, dt=dt, ft=ft, n=n, trscheme=trscheme) 
        end
    else
        gf = zeros(ComplexF64, half, half, length(0:dt:ft))
        idx = Threads.Atomic{Int}(1)
        n = half
        Threads.@sync for _ in 1:parallel
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1) 
                i > n && break  
                gf[i,:,:] = propagator(gs, H, op, i; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=rev, dt=dt, ft=ft, n=n, trscheme=trscheme) 
            end
        end
    end
    for i in eachindex(t)
        factor = rev ? -im*exp(-im*gsenergy*t[i]) : -im*exp(im*gsenergy*t[i])
        gf[:,:,i] = factor*gf[:,:,i]
    end
    return gf
end

"""
    dcorrelator(::Type{R}, H::MPOHamiltonian, gsenergy::Number, mps::Vector{<:FiniteMPS})
"""
function dcorrelator(::Type{R}, H::MPOHamiltonian, gsenergy::Number, mps::Vector{<:FiniteMPS}; 
                    parallel::Union{String, Integer}=Threads.nthreads(), 
                    verbose=true, 
                    path::String="./", 
                    dt::Number=0.05, 
                    ft::Number=5.0, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3)) where R<:RetardedGF
    t, half = collect(0:dt:ft), length(mps)÷2
    if parallel == "np"
        gf = SharedArray{ComplexF64, 3}(length(mps), half, length(0:dt:ft))
        @sync @distributed for i in 1:length(mps)
            if i <= half
                gf[i,:,:] = propagator(H, mps[1:half], mps[i]; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
            else
                gf[i,:,:] = propagator(H, mps[(half+1):end], mps[i]; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme)
            end
        end
    elseif parallel == 1
        gf = zeros(ComplexF64, length(mps), half, length(0:dt:ft))
        for i in 1:length(mps)
            if i <= half
                gf[i,:,:] = propagator(H, mps[1:half], mps[i]; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
            else
                gf[i,:,:] = propagator(H, mps[(half+1):end], mps[i]; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme)
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
                    gf[i,:,:] = propagator(H, mps[1:half], mps[i]; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
                else
                    gf[i,:,:] = propagator(H, mps[(half+1):end], mps[i]; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme)
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

"""
    dcorrelator(::Type{R}, gf_slices::AbstractArray{<:AbstractMatrix}, gsenergy::Number, t::AbstractRange) where R<:RetardedGF
"""
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


"""
    dcorrelator(::Type{GreaterLessGF}, gs::AbstractFiniteMPS, H::MPOHamiltonian, ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap})
"""
function dcorrelator(::Type{GreaterLessGF}, gs::AbstractFiniteMPS, H::MPOHamiltonian, ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap}; 
    whichs=:both, 
    verbose=true, 
    path::String="./", 
    savekets=false,  
    dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncerr(1e-3))
    t, half = collect(0:dt:ft), length(H)
    gsenergy = expectation_value(gs, H)
    if whichs == :both
        gf = SharedArray{ComplexF64, 3}(2*half, half, length(0:dt:ft))
        @sync @distributed for i in 1:(2*half)
            if i <= half
                gf[i,:,:] = propagator(gs, H, ops[1], i; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
            else
                gf[i,:,:] = propagator(gs, H, ops[2], i; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme)
            end
        end
        for i in eachindex(t)
            factor₁, factor₂ = -im*exp(im*gsenergy*t[i]), -im*exp(-im*gsenergy*t[i])
            gf[1:half,:,i] = factor₁*gf[1:half,:,i]
            gf[(half+1):end,:,i] = factor₂*gf[(half+1):end,:,i]
        end
        return [gf[1:half,:,:], gf[(half+1):end,:,:]]
    elseif whichs == :greater
        gf = SharedArray{ComplexF64, 3}(half, half, length(0:dt:ft))
        @sync @distributed for i in 1:half
            gf[i,:,:] = propagator(gs, H, ops[1], i; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
        end
        for i in eachindex(t)
            gf[:,:,i] = -im*exp(im*gsenergy*t[i])*gf[:,:,i]
        end
        return gf
    elseif whichs == :less
        gf = SharedArray{ComplexF64, 3}(half, half, length(0:dt:ft))
        @sync @distributed for i in 1:half
            gf[i,:,:] = propagator(gs, H, ops[2], i; filename=joinpath(path, "gf_slice_$(i)_$(dt)_$(ft).jld2"), verbose=verbose, savekets=savekets, rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme) 
        end
        for i in eachindex(t)
            gf[:,:,i] = -im*exp(-im*gsenergy*t[i])*gf[:,:,i]
        end
        return gf
    end
end
