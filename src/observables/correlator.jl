"""
    propagator(H::MPOHamiltonian, bra::FiniteMPS, ket::FiniteMPS; rev::Bool=false, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncdim(50))
    propagator(H::MPOHamiltonian, bras::Vector{<:FiniteMPS}, ket::FiniteMPS; rev::Bool=false, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncdim(50))
"""
function propagator(H::MPOHamiltonian, bra::FiniteMPS, ket::FiniteMPS; rev::Bool=false, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncdim(50))
    times = collect(0:dt:ft)
    propagators = zeros(ComplexF64, length(times))
    propagators[1] = dot(bra, ket)
    envs = environments(ket, H)
    for (i, t) in enumerate(times[2:end])
        alg = t > n * dt ? TDVP() : TDVP2(; trscheme=trscheme)
        ket, envs = timestep(ket, H, 0, dt, alg, envs)
        propagators[i+1] = dot(bra, ket)
    end
    rev ? propagators = conj.(propagators) : propagators = propagators
    return propagators
end

function propagator(H::MPOHamiltonian, bras::Vector{<:FiniteMPS}, ket::FiniteMPS; rev::Bool=false, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncdim(50))
    times = collect(0:dt:ft)
    propagators = zeros(ComplexF64, length(bras), length(times))
    propagators[:,1] = [dot(bras[i], ket) for i in 1:length(bras)]
    envs = environments(ket, H)
    for (i, t) in enumerate(times[2:end])
        alg = t > n * dt ? TDVP() : TDVP2(; trscheme=trscheme)
        ket, envs = timestep(ket, H, 0, dt, alg, envs)
        for j in eachindex(bras)
            propagators[j,i+1] = dot(bras[j], ket)
        end
    end
    rev ? propagators = conj.(propagators) : propagators = propagators
    return propagators
end

struct RetardedGF{K} end
RetardedGF(::Type{RetardedGF{:f}}) = 1
RetardedGF(::Type{RetardedGF{:b}}) = -1

struct GreaterGF end
struct LesserGF end
struct GorkovGF end
struct MatsubaraGF end

function dcorrelator(::Type{R}, H::MPOHamiltonian, groundenergy::Number, mps::Vector{<:FiniteMPS}; dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncdim(50)) where R<:RetardedGF
    factor₁, factor₂, half = exp(im*groundenergy*dt), RetardedGF(R)*exp(-im*groundenergy*dt), length(mps)÷2
    gf = SharedArray{ComplexF64, 3}(length(mps), half, length(0:dt:ft))
    @sync @distributed for i in 1:length(mps)
        if i <= half
            gf[i,:,:] = propagator(H, mps[1:half], mps[i]; rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
        else
            gf[i,:,:] = propagator(H, mps[(half+1):end], mps[i]; rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme)
        end
    end
    return factor₁*gf[1:half,:,:] + factor₂*gf[(half+1):end,:,:]
end

