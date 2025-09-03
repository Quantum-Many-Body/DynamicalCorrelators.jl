"""
    chargedMPS(operator::AbstractTensorMap, state::FiniteMPS, site::Integer)
"""
function chargedMPS(operator::AbstractTensorMap, state::FiniteMPS, site::Integer)
    return chargedMPO(operator, site, length(state))*state
end

function chargedMPS(operator::AbstractTensorMap, state::FiniteMPS, site₁::Integer, site₂::Integer)
    return chargedMPO(operator, site₁, site₂, length(state))*state
end

"""
    charged super MPS
"""
const FiniteSuperMPS{C} = FiniteMPS{<:AbstractTensorMap{N,C,3,1}} where {N}

function chargedMPS(op::AbstractTensorMap{B,S,1,2}, mps::FiniteSuperMPS, site::Integer) where {B, S}
    T = promote_contract(scalartype(op), scalartype(mps))
    A = similarstoragetype(eltype(mps), T)
    A2 = map(1:length(mps)) do i
        A1 = i == 1 ? mps.AC[1] : mps.AR[i]
        if i < site
            a = A1
        end
        if i == site 
            F = fuser(A, domain(A1, 1), domain(op, 2))
            @plansor a[-1 -2 -3; -4] := A1[-1 1 3; 5] * op[-2; 1 2] * τ[2 -3; 3 4] * conj(F[-4; 5 4])
        end
        if i > site
            Fl, Fr = fuser(A, codomain(A1, 1), domain(op, 2)), fuser(A, domain(A1, 1), domain(op, 2))
            @plansor a[-1 -2 -3; -4] := Fl[-1; 1 2] * A1[1 3 5; 7] * τ[2 -2; 3 4] * τ[4 -3; 5 6] * conj(Fr[-4; 7 6])
        end
        return a
    end
    trscheme = truncbelow(eps(real(T)))
    return changebonds!(FiniteMPS(A2), SvdCut(; trscheme); normalize = false)
end

function chargedMPS(op::AbstractTensorMap{S,B,2,1}, mps::FiniteSuperMPS, site::Integer) where {B, S}
    T = promote_contract(scalartype(op), scalartype(mps))
    A = similarstoragetype(eltype(mps), T)
    A2 = map(1:length(mps)) do i
        A1 = i == 1 ? mps.AC[1] : mps.AR[i]
        if i < site
            Fl, Fr = fuser(A, codomain(A1, 1), codomain(op, 1)), fuser(A, domain(A1, 1), codomain(op, 1))
            @plansor a[-1 -2 -3; -4] := Fl[-1; 1 2] * A1[1 3 5; 7] * τ[2 -2; 3 4] * τ[4 -3; 5 6] * conj(Fr[-4; 7 6])
        end
        if i == site 
            F = fuser(A, codomain(A1, 1), codomain(op, 1))
            @plansor a[-1 -2 -3; -4] := F[-1; 1 2] * A1[1 3 -3; -4] * op[2 -2; 3]
        end
        if i > site
            a = A1
        end
        return a
    end
    trscheme = truncbelow(eps(real(T)))
    return changebonds!(FiniteMPS(A2), SvdCut(; trscheme); normalize = false)
end

function chargedMPS(op::AbstractTensorMap{S,B,1,1}, mps::FiniteSuperMPS, site::Integer) where {B, S}
    T = promote_contract(scalartype(op), scalartype(mps))
    A2 = map(1:length(mps)) do i
        A1 = i == 1 ? mps.AC[1] : mps.AR[i]
        if i !== site
            a = A1
        else
            @plansor a[-1 -2 -3; -4] := A1[-1 1 -3; -4] * op[-2; 1]
        end
        return a
    end
    trscheme = truncbelow(eps(real(T)))
    return changebonds!(FiniteMPS(A2), SvdCut(; trscheme); normalize = false)
end
