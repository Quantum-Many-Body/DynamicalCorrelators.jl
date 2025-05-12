function broaden_gauss(eta::Real, t::Real)
    return exp(-(eta*t)^2)
end

function broaden_lorentz(eta::Real, t::Real)
    return exp(-eta*abs(t))/π
end

function fourier_kw(gf_rt::AbstractArray, rs::AbstractArray{<:AbstractArray}, ts::AbstractRange, k::AbstractArray{<:Number}, w::Number; broadentype::String="G", eta::Real=0.05, regroup::AbstractArray{<:AbstractArray}=[Vector(1:size(gf_rt,1)),])
    @assert size(gf_rt, 1) == length(rs) "Dimension mismatch: the length of site positions 'rs' must equal to the size of green function matrix 'gf_rt'!"
    if broadentype == "G"
        broaden = broaden_gauss
    elseif broadentype == "L"
        broaden = broaden_lorentz
    else
        throw(ArgumentError("Invalid broadening type: $broadentype"))
    end
    dest = zeros(ComplexF64, length(regroup), length(regroup))
    for x in eachindex(regroup), y in eachindex(regroup), l in eachindex(ts)
        for j in eachindex(regroup[x]), i in eachindex(regroup[y])
            dest[x, y] += gf_rt[regroup[x][j], regroup[y][i], l]*cis(-dot(k, rs[regroup[y][i]]-rs[regroup[x][j]])+w*ts[l])*broaden(eta, ts[l])
        end
    end
    return dest*(ts.step.hi)/length(regroup[1])/4π^2
end

function fourier_kw(gf_rt::AbstractArray, rs::AbstractArray{<:AbstractArray}, ts::AbstractRange, ks::AbstractArray{<:AbstractArray}, ws::AbstractArray{<:Number}; mthreads::Integer=Threads.nthreads(), kwargs...)
    gf_kw = Matrix(undef, length(ws), length(ks))
    if mthreads == 1
        for k in eachindex(ks), w in eachindex(ws)
            gf_kw[w, k] = fourier_kw(gf_rt, rs, ts, ks[k], ws[w]; kwargs...)
        end
    else
        idx = Threads.Atomic{Int}(1)
        indices = CartesianIndices((length(ks), length(ws)))
        n = length(indices)
        Threads.@sync for _ in 1:mthreads
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1)  
                i > n && break  
                k, w = indices[i].I
                gf_kw[w, k] = fourier_kw(gf_rt, rs, ts, ks[k], ws[w]; kwargs...)
            end
        end
    end
    return gf_kw
end

function fourier_rw(gf_rt::AbstractArray, ts::AbstractRange, ws::AbstractArray{<:Number}; broadentype::String="G", eta::Real=0.05, mthreads::Integer=Threads.nthreads())
    if broadentype == "G"
        broaden = broaden_gauss
    elseif broadentype == "L"
        broaden = broaden_lorentz
    else
        throw(ArgumentError("Invalid broadening type: $broadentype"))
    end
    gf_rw = zeros(ComplexF64, size(gf_rt, 1), size(gf_rt, 1), length(ws))
    if mthreads == 1
        for i in eachindex(ws)
            for j in eachindex(ts)
                gf_rw[:,:,i] .+= gf_rt[:,:,j]*exp(im*ws[i]*ts[j])*broaden(eta, ts[j])
            end
        end
    else
        idx = Threads.Atomic{Int}(1)
        n = length(ws)
        Threads.@sync for _ in 1:mthreads
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1)  
                i > n && break  
                for j in eachindex(ts)
                    gf_rw[:,:,i] .+= gf_rt[:,:,j]*exp(im*ws[i]*ts[j])*broaden(eta, ts[j])
                end
            end
        end
    end
    return permutedims(gf_rw, (2,1,3))*(ts.step.hi)
end