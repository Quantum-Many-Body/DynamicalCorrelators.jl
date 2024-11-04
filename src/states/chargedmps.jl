"""
    chargedMPS(operator::TensorMap, state::FiniteMPS, site::Integer)
"""
function chargedMPS(operator::TensorMap, state::FiniteMPS, site::Integer)
    pspace = domain(operator)[1]
    if (length(domain(operator)) == 2)&&(length(codomain(operator)) == 1)
        Z, vspace = fZ(operator), domain(operator)[1]
        I = isomorphism(storagetype(operator), oneunit(vspace)*pspace, pspace*oneunit(vspace))
        mpo = FiniteMPO([i < site ? I : i == site ? add_single_util_leg(operator) : Z for i in 1:length(state)])
    elseif (length(codomain(operator)) == 2)&&(length(domain(operator)) == 1)
        Z, vspace = fZ(operator), codomain(operator)[1]
        I = isomorphism(storagetype(operator), oneunit(vspace)*pspace, pspace*oneunit(vspace))
        mpo = FiniteMPO([i < site ? Z : i == site ? add_single_util_leg(operator) : I for i in 1:length(state)])
    elseif (length(codomain(operator)) == 1)&&(length(domain(operator)) == 1)
        I = add_util_leg(isomorphism(storagetype(operator), pspace, pspace))
        mpo = FiniteMPO([i < site ? I : i == site ? add_util_leg(operator) : I for i in 1:length(state)])
    else
        throw(ArgumentError("invalid creation or annihilation operator"))
    end
    return mpo*state
end

