"""
    chargedMPO(operator::AbstractTensorMap, site::Integer, nsites::Integer)
"""
function chargedMPO(operator::AbstractTensorMap, site::Integer, nsites::Integer)
    Z, pspace = fZ(operator), domain(operator)[1]
    if (length(domain(operator)) == 2)&&(length(codomain(operator)) == 1)
        vspace = domain(operator)[2]
        I = isomorphism(storagetype(operator), oneunit(vspace)*pspace, pspace*oneunit(vspace))
        mpo = FiniteMPO([i < site ? I : i == site ? add_single_util_leg(operator) : Z for i in 1:nsites])
    elseif (length(codomain(operator)) == 2)&&(length(domain(operator)) == 1)
        vspace = codomain(operator)[1]
        I = isomorphism(storagetype(operator), oneunit(vspace)*pspace, pspace*oneunit(vspace))
        mpo = FiniteMPO([i < site ? Z : i == site ? add_single_util_leg(operator) : I for i in 1:nsites])
    elseif (length(codomain(operator)) == 1)&&(length(domain(operator)) == 1)
        I = add_util_leg(isomorphism(storagetype(operator), pspace, pspace))
        mpo = FiniteMPO([i < site ? I : i == site ? add_util_leg(operator) : I for i in 1:nsites])
    else
        throw(ArgumentError("invalid operator, expected 2-leg or 3-leg tensor"))
    end
    return mpo
end