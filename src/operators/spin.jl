
function S_plus(elt::Type{<:Number}, ::Type{SU2Irrep}; side=:L)
    pspace = SU2Space(1//2 => 1)
    vspace = SU2Space(1 => 1)
    if side == :L
        sp = TensorMap(ones, elt, pspace, pspace ⊗ vspace) * sqrt(3) / 2
    end
    return sp
end

function S_min(elt::Type{<:Number}, ::Type{SU2Irrep}; side=:R)
    if side == :R
        sm = permute(S_plus(elt, SU2Irrep; side=:L)', ((2, 1), (3,)))
    end
    return sm
end

function heisenberg(elt::Type{<:Number}, ::Type{SU2Irrep})
    return contract_twosite(S_plus(elt, SU2Irrep; side=:L), S_min(elt, SU2Irrep; side=:R))
end