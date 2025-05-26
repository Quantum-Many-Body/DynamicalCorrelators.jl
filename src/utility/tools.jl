function add_single_util_leg(tensor::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    ou = oneunit(_firstspace(tensor))
    if length(codomain(tensor)) < length(domain(tensor))
        util = isomorphism(storagetype(tensor), ou * codomain(tensor), codomain(tensor))
        tensors = util * tensor
    elseif length(codomain(tensor)) > length(domain(tensor))
        util = isomorphism(storagetype(tensor), domain(tensor), domain(tensor) * ou)
        tensors = tensor * util
    else 
        throw(ArgumentError("invalid operator"))
    end
    return tensors
end

function cart2polar(point::AbstractArray)
    r = norm(point) 
    θ = atan(point[2], point[1])
    ϕ = length(point) == 3 ? acos(point[3]/r) : π/2
    return r, θ, ϕ
end

function phase_by_polar(theta::AbstractVector, phi::AbstractVector, phases::AbstractVector)
    function _phase_by_polar(bond::Bond)
        _, θ, ϕ = cart2polar(rcoordinate(bond))
        for i in eachindex(phases)
            any(≈(θ), theta[i]) && any(≈(ϕ), phi[i]) && return phases[i]
        end
    end
    return _phase_by_polar
end