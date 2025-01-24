"""
    chargedMPO(operator::AbstractTensorMap, site::Integer, nsites::Integer)
"""
function chargedMPO(operator::AbstractTensorMap, site::Integer, nsites::Integer)
    pspace = domain(operator)[1]
    if (length(domain(operator)) == 2)&&(length(codomain(operator)) == 1)
        Z, vspace = fZ(operator), domain(operator)[2]
        I = isomorphism(storagetype(operator), oneunit(vspace)*pspace, pspace*oneunit(vspace))
        mpo = FiniteMPO([i < site ? I : i == site ? add_single_util_leg(operator) : Z for i in 1:nsites])
    elseif (length(codomain(operator)) == 2)&&(length(domain(operator)) == 1)
        Z, vspace = fZ(operator), codomain(operator)[1]
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

function chargedMPO(operator₁::AbstractTensorMap, operator₂::AbstractTensorMap, site₁::Integer, site₂::Integer, nsites::Integer)
    pspace = domain(operator₁)[1]
    if (length(domain(operator₁)) == 2)&&(length(codomain(operator₁)) == 1)&&(length(codomain(operator₂)) == 2)&&(length(domain(operator₂)) == 1)&&(site₁ < site₂)
        Z, vspace = fZ(operator₁), domain(operator₁)[2]
        I = isomorphism(storagetype(operator₁), oneunit(vspace)*pspace, pspace*oneunit(vspace))
        mpo = FiniteMPO([i < site₁ ? I : i == site₁ ? add_single_util_leg(operator₁) : (site₁ < i < site₂) ? Z : i == site₂ ? add_single_util_leg(operator₂) : I for i in 1:nsites])
    else
        throw(ArgumentError("invalid operator, expected operator₁ at left and operator₂ at right"))
    end
    return mpo
end