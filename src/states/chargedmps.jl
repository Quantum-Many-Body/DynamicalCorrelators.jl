"""
    chargedMPS(operator::AbstractTensorMap, state::FiniteMPS, site::Integer)
"""
function chargedMPS(operator::AbstractTensorMap, state::FiniteMPS, site::Integer)
    mpo = chargedMPO(operator, site, length(state))
    return mpo*state
end

function chargedMPS(H::MPOHamiltonian, operator::AbstractTensorMap, state::FiniteMPS, site::Integer; imag::Bool=false, dt::Number=0.05, ft::Number=5.0, n::Integer=3, trscheme=truncerr(1e-3))
    mps = chargedMPO(operator, site, length(state))*state
    times = collect(0:dt:ft)
    mpsset = Vector(undef, length(times))
    mpsset[1] = mps
    envs = environments(mps, H)
    if imag
        for (i, t) in enumerate(times[2:end])
            alg = t > n * dt ? TDVP() : TDVP2(; trscheme=trscheme)
            mps, envs = timestep(mps, H, 0, -1im*dt, alg, envs)
            mpsset[i+1] = mps
        end
    else
        for (i, t) in enumerate(times[2:end])
            alg = t > n * dt ? TDVP() : TDVP2(; trscheme=trscheme)
            mps, envs = timestep(mps, H, 0, dt, alg, envs)
            mpsset[i+1] = mps
        end
    end
    return mpsset
end