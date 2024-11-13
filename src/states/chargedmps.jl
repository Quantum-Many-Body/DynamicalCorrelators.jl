"""
    chargedMPS(operator::AbstractTensorMap, state::FiniteMPS, site::Integer)
"""
function chargedMPS(operator::AbstractTensorMap, state::FiniteMPS, site::Integer)
    mpo = chargedMPO(operator, site, length(state))
    return mpo*state
end
