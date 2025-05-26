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

function sort_by_distance(latt::CustomLattice, ij)
    is = ij[length(ij)÷2:-1:1]
    js = ij[length(ij)÷2+1:1:length(ij)]
    ks = is[2:end]
    ls = js[1:end-1]
    r = []
    for i in eachindex(is)
        push!(r, [is[i], js[i], norm(latt.lattice[is[i]]-latt.lattice[js[i]])])
    end
    for i in eachindex(ks)
        push!(r, [ks[i], ls[i], norm(latt.lattice[ks[i]]-latt.lattice[ls[i]])])
    end
    sort!(r, by=x->x[3])
    is = zeros(Int64, length(r))
    js = zeros(Int64, length(r))
    ds = zeros(Float64, length(r))
    for i in eachindex(is)
        is[i] = Int(r[i][1])
        js[i] = Int(r[i][2])
        ds[i] = r[i][3]
    end
    return is, js, ds
end

