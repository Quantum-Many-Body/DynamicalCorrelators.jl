# t = 0:Δt:T, w_max < π/Δt, Δw < 2π/T (Δw≈π/2T)

function broaden_gauss(t::Real, eta::Real)
    return exp(-(eta*t)^2)
end

function broaden_lorentz(t::Real, eta::Real)
    return exp(-eta*abs(t))
end

function blackman_window(t::Real, T::Real)
    return 0.42 - 0.5*cos(2π*t/T) + 0.08*cos(4π*t/T)
end

function parzen_window(t::Real, T::Real)
	if abs(t/T) < 1/2
		return 6*abs(t/T)^3 - 6*(t/T)^2 + 1
	elseif abs(t/T) < 1
		return 2*(1 - abs(t/T))^3
	else
		return 0.0
	end
end

function damping(t, broadentype)
    if broadentype[2] == "G"
        broaden = broaden_gauss
    elseif broadentype[2] == "L"
        broaden = broaden_lorentz
    elseif broadentype[2] == "B"
        broaden = blackman_window
    elseif broadentype[2] == "P"
        broaden = parzen_window
    else
        throw(ArgumentError("Invalid broadening type: $broadentype"))
    end
    return broaden(t, broadentype[1])
end

function fourier_kt(gf_rt::AbstractArray, rs::AbstractArray{<:AbstractArray}, k::AbstractArray{<:Number}; regroup::AbstractArray{<:AbstractArray}=[Vector(1:size(gf_rt,1)),])
    dest = zeros(ComplexF64, length(regroup), length(regroup), size(gf_rt, 3))
    for x in eachindex(regroup), y in eachindex(regroup), l in axes(gf_rt, 3)
        for j in eachindex(regroup[x]), i in eachindex(regroup[y])
            dest[x, y, l] += gf_rt[regroup[x][j], regroup[y][i], l]*cis(-dot(k, rs[regroup[y][i]]-rs[regroup[x][j]]))
        end
    end
    return dest
end

function fourier_kw(gf_kt::AbstractArray, ts::AbstractRange, w::Number, dampings::AbstractArray)
    dest = zeros(ComplexF64, size(gf_kt, 1), size(gf_kt, 2))
    for x in axes(gf_kt, 1), y in axes(gf_kt, 2)
        temp = gf_kt[x, y, :] .* cis.(w*ts) .* dampings
        dest[x, y] = integrate(ts, temp) 
    end
    return dest
end

function fourier_kw(gf_rt::AbstractArray, rs::AbstractArray{<:AbstractArray}, ts::AbstractRange, ks::AbstractArray{<:AbstractArray}, ws::AbstractArray{<:Number}; 
                    mthreads::Integer=Threads.nthreads(), broadentype=(0.05, "G"), regroup::AbstractArray{<:AbstractArray}=[Vector(1:size(gf_rt,1)),])
    @assert size(gf_rt, 1) == length(rs) "Dimension mismatch: the length of site positions 'rs' must equal to the size of green function matrix 'gf_rt'!"
    dampings = [damping(t, broadentype) for t in ts] 
    gf_kw = Matrix(undef, length(ws), length(ks))
    for k in eachindex(ks)
        gf_kt = fourier_kt(gf_rt, rs, ks[k]; regroup=regroup)
        idx = Threads.Atomic{Int}(1)
        Threads.@sync for _ in 1:mthreads
            Threads.@spawn while true
                w = Threads.atomic_add!(idx, 1)  
                w > length(ws) && break  
                gf_kw[w, k] = fourier_kw(gf_kt, ts, ws[w], dampings)
            end
        end
    end
    return gf_kw/(4π^2)
end

function fourier_rw(gf_rt::AbstractArray, ts::AbstractArray, ws::AbstractArray; broadentype=(0.05, "G"), mthreads::Integer=Threads.nthreads(), ifsum::Bool=true)
    dampings = [damping(t, broadentype) for t in ts] 
    gf_rw = zeros(ComplexF64, size(gf_rt, 1), size(gf_rt, 1), length(ws))
    if mthreads == 1
        for i in eachindex(ws)
            for a in axes(gf_rt, 1)
                for b in axes(gf_rt, 2)
                    temp = gf_rt[a,b,:] .* cis.(ws[i]*ts).* dampings
                    gf_rw[a,b,i] = ifsum ? sum(temp)*((ts[end]-ts[1])/length(ts)) : integrate(ts, temp)
                end
            end
        end
    else
        idx = Threads.Atomic{Int}(1)
        n = length(ws)
        Threads.@sync for _ in 1:mthreads
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1)  
                i > n && break  
                for a in axes(gf_rt, 1)
                    for b in axes(gf_rt, 2)
                        temp = gf_rt[a,b,:] .* cis.(ws[i]*ts).* dampings
                        gf_rw[a,b,i] = ifsum ? sum(temp)*((ts[end]-ts[1])/length(ts)) : integrate(ts, temp)
                    end
                end
            end
        end
    end
    return gf_rw
end

function static_structure_factor(ss, rs, ks)
    sf = zeros(ComplexF64, length(ks))
    for i in eachindex(ks)
        for a in eachindex(rs), b in eachindex(rs)
            sf[i] += ss[a, b]*cis(dot(ks[i], rs[a]-rs[b]))
        end
    end
    return sf/length(rs)
end