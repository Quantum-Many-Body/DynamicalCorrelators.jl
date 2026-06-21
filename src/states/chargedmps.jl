"""
    const FiniteNormalMPS{C}

Type alias for a standard finite MPS with 2-leg physical tensors (virtual-physical; virtual).
Used for zero-temperature ground states.
"""
const FiniteNormalMPS{C} = FiniteMPS{<:AbstractTensorMap{N,C,2,1}} where {N}

"""
    const FiniteSuperMPS{C}

Type alias for a finite MPS with 3-leg physical tensors (virtual-physical-ancilla; virtual).
Used for finite-temperature density matrix purification (thermofield double formalism).
"""
const FiniteSuperMPS{C} = FiniteMPS{<:AbstractTensorMap{N,C,3,1}} where {N}

"""
    chargedMPS(operator::AbstractTensorMap, state::FiniteNormalMPS, site::Integer)

Apply a local operator at `site` to a normal MPS by constructing a charged MPO and
contracting it with the state. Returns the resulting MPS (e.g., `c†ᵢ|ψ⟩`).
"""
function chargedMPS(operator::AbstractTensorMap, state::FiniteNormalMPS, site::Integer)
    return chargedMPO(operator, site, length(state))*state
end

"""
    chargedMPS(operator::AbstractTensorMap, state::FiniteNormalMPS, site₁::Integer, site₂::Integer)

Apply a two-site operator at `site₁` and `site₂` to a normal MPS.
"""
function chargedMPS(operator::AbstractTensorMap, state::FiniteNormalMPS, site₁::Integer, site₂::Integer)
    return chargedMPO(operator, site₁, site₂, length(state))*state
end

"""
    chargedMPS(op::AbstractTensorMap{B,S,1,2}, mps::FiniteSuperMPS, site::Integer)

Apply a (1,2)-leg operator (1 codomain, 2 domain legs) to a super MPS at `site`.
This inserts the operator into the physical-ancilla leg structure of the purified state.
The operator is assumed to use the `side=:L` convention. Sites at and after the
operator site use materialized `TensorMap(BraidingTensor(...))` tensors for the
fermionic string, so the propagated string has a fixed braiding channel.
"""
function chargedMPS(op::AbstractTensorMap{B,S,1,2}, mps::FiniteSuperMPS, site::Integer) where {B, S}
    T = promote_contract(scalartype(op), scalartype(mps))
    A = similarstoragetype(eltype(mps), T)
    A2 = map(1:length(mps)) do i
        A1 = i == 1 ? mps.AC[1] : mps.AR[i]
        if i < site
            a = A1
        end
        if i == site 
            τ2 = TensorMap(BraidingTensor(dual(codomain(A1,2)), domain(op, 2)))
            F = fuser(A, domain(A1, 1), domain(op, 2))
            @plansor a[-1 -2 -3; -4] := A1[-1 1 3; 5] * op[-2; 1 2] * τ2[2 -3; 3 4] * conj(F[-4; 5 4])
        end
        if i > site
            τ1 = TensorMap(BraidingTensor(codomain(A1,2), domain(op, 2)))
            τ2 = TensorMap(BraidingTensor(dual(codomain(A1,2)), domain(op, 2)))
            Fl, Fr = fuser(A, codomain(A1, 1), domain(op, 2)), fuser(A, domain(A1, 1), domain(op, 2))
            @plansor a[-1 -2 -3; -4] := Fl[-1; 1 2] * A1[1 3 5; 7] * τ1[2 -2; 3 4] * τ2[4 -3; 5 6] * conj(Fr[-4; 7 6])
        end
        return a
    end
    trscheme = trunctol(; atol = eps(real(T)))
    return changebonds!(FiniteMPS(A2), SvdCut(; trscheme); normalize = false)
end

"""
    chargedMPS(op::AbstractTensorMap{S,B,1,1}, mps::FiniteSuperMPS, site::Integer)

Apply a (1,1)-leg operator (diagonal, charge-neutral) to a super MPS at `site`.
Only the tensor at the operator site is modified; all other tensors remain unchanged.
"""
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
    trscheme = trunctol(; atol = eps(real(T)))
    return changebonds!(FiniteMPS(A2), SvdCut(; trscheme); normalize = false)
end

"""
    chargedMPS(op::AbstractTensorMap, gs::AbstractFiniteMPS, site::Integer, alg)

Approximate `chargedMPO(op, site, length(gs)) * gs` with the supplied MPSKit
algorithm `alg`.

The initial state is a random finite MPS in the charge sector implied by `op`
and with internal bond spaces inherited from `gs`. This is useful for charged
states whose exact MPO application would create an inconvenient bond
dimension.
"""
function chargedMPS(op::AbstractTensorMap, gs::AbstractFiniteMPS, site::Integer, alg)
    ψ, = approximate(randFiniteMPS(eltype(gs[1]), gs, op), (chargedMPO(op, site, length(gs)), gs), alg)
    return ψ
end

"""
    identityMPS(H::FiniteMPOHamiltonian)

Construct an identity MPS (purified infinite-temperature density matrix) from a Hamiltonian.
Each site tensor is a `BraidingTensor` acting as an identity map on the physical space.
This serves as the initial state for imaginary-time evolution to finite temperature.
"""
function identityMPS(H::FiniteMPOHamiltonian)
    V = oneunit(spacetype(H))
    W = map(1:length(H)) do site
        return BraidingTensor{scalartype(H)}(physicalspace(H, site), V)
    end
    return convert(FiniteMPS, FiniteMPO(W))
end
