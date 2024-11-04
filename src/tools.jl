function add_single_util_leg(tensor::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    ou = oneunit(_firstspace(tensor))
    if (length(codomain(tensor))==1)&&(length(domain(tensor))==2)
        util = isomorphism(storagetype(tensor), ou * codomain(tensor), codomain(tensor))
        fourlegtensor = util * tensor
    elseif (length(codomain(tensor))==2)&&(length(domain(tensor))==1)
        util = isomorphism(storagetype(tensor), domain(tensor), domain(tensor) * ou)
        fourlegtensor = tensor * util
    else 
        throw(ArgumentError("invalid creation or annihilation operator"))
    end
    return fourlegtensor
end

function setprocs(numworkers::Int)
    np = length(workers())
    np < numworkers ? addworkers=(numworkers - np) : addworkers = 0
    addprocs(addworkers)
end