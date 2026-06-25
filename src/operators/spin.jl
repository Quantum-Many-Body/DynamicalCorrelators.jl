"""
    S_plus(elt::Type{<:Number}, ::Type{SU2Irrep}, spin; side=:L)
"""
function S_plus(elt::Type{<:Number}, ::Type{SU2Irrep}, spin; side=:L)
    pspace = SU2Space(spin => 1)
    vspace = SU2Space(1 => 1)
    if side == :L
        sp = ones(elt, pspace, pspace ⊗ vspace) * sqrt(spin^2+spin)
    elseif side == :R
        E = S_plus(elt, SU2Irrep, spin; side=:L)
        vspace = domain(E, 2)
        F = isomorphism(storagetype(E), vspace, flip(vspace))
        @planar sp[-1 -2; -3] := E[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return sp
end

"""
    S_min(elt::Type{<:Number}, ::Type{SU2Irrep}, spin; side=:R)
"""
function S_min(elt::Type{<:Number}, ::Type{SU2Irrep}, spin; side=:R)
    if side == :R
        sm = permute(S_plus(elt, SU2Irrep, spin; side=:L)', ((2, 1), (3,)))
    elseif side == :L
        E = S_plus(elt, SU2Irrep, spin; side=:L)'
        F = isomorphism(storagetype(E), flip(space(E, 2)), space(E, 2))
        @planar sm[-1; -2 -3] := E[-1 1; -2] * F[-3; 1]    
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return sm
end

"""
    heisenberg(elt::Type{<:Number}, ::Type{SU2Irrep}, spin::Number)
"""
function heisenberg(elt::Type{<:Number}, ::Type{SU2Irrep}, spin::Number)
    return contract_twosite(S_plus(elt, SU2Irrep, spin; side=:L), S_min(elt, SU2Irrep, spin; side=:R))
end