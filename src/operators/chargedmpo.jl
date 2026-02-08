"""
    chargedMPO(operator::AbstractTensorMap, site::Integer, nsites::Integer)

Construct a `FiniteMPO` that applies a single-site operator at position `site` on a chain
of `nsites` sites. A fermionic string operator `fZ` (Jordan-Wigner string) is inserted on
sites to the left or right of `site` depending on the operator leg structure:
- (1 codomain, 2 domain): creation-like → string on the left.
- (2 codomain, 1 domain): annihilation-like → string on the right.
- (1, 1): diagonal → identity everywhere else.
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

"""
    chargedMPO(operator₁, operator₂, site₁, site₂, nsites)

Construct a `FiniteMPO` that applies two operators at positions `site₁ < site₂`.
Requires `operator₁` to be creation-like (1,2) and `operator₂` to be annihilation-like (2,1).
A Jordan-Wigner string is inserted between the two sites.
"""
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

"""
    chargedMPO(operator::AbstractTensorMap, site₁, site₂, nsites)

Construct a `FiniteMPO` for a two-site operator by decomposing it via SVD into
local tensors at `site₁` and `site₂`, with appropriate string operators in between.
The decomposition handles creation-like, annihilation-like, and diagonal operators.
"""
function chargedMPO(operator::AbstractTensorMap, site₁::Integer, site₂::Integer, nsites::Integer)
    pspace = domain(operator)[1]
    O₁, O₂ = decompose_localmpo(add_single_util_leg(operator))
    iso₁ = isomorphism(storagetype(operator), codomain(O₂)[1], codomain(O₂)[1])
    iso₂ = isomorphism(storagetype(operator), pspace, pspace)
    @planar Z₁[-1 -2; -3 -4] := iso₁[-1; 1] * iso₂[-2; 2] * τ[1 2; 3 4] * iso₂[3; -3] * iso₁[4; -4]
    Z₂ = fZ(operator)
    I = isomorphism(storagetype(operator), oneunit(pspace)*pspace, pspace*oneunit(pspace))
    if length(domain(operator)) > length(codomain(operator))
        mpo = FiniteMPO([i < site₁ ? I : i == site₁ ? O₁ : site₁ < i < site₂ ? Z₁ : i == site₂ ? O₂ : Z₂ for i in 1:nsites])
    elseif length(codomain(operator)) > length(domain(operator))
        mpo = FiniteMPO([i < site₁ ? Z₂ : i == site₁ ? O₁ : site₁ < i < site₂ ? Z₁ : i == site₂ ? O₂ : I for i in 1:nsites])
    elseif length(codomain(operator)) == length(domain(operator))
        I = add_util_leg(isomorphism(storagetype(operator), pspace, pspace))
        mpo = FiniteMPO([i < site₁ ? I : i == site₁ ? O₁ : site₁ < i < site₂ ? Z₁ : i == site₂ ? O₂ : I for i in 1:nsites])
    else
        throw(ArgumentError("invalid operator"))
    end
    return mpo
end

"""
    identityMPO(H::FiniteMPOHamiltonian)

Construct an identity `FiniteMPO` compatible with Hamiltonian `H`.
Each site is a `BraidingTensor` acting as an identity on the physical space.
"""
function identityMPO(H::FiniteMPOHamiltonian)
    V = oneunit(spacetype(H))
    W = map(1:length(H)) do site
        return BraidingTensor{scalartype(H)}(physicalspace(H, site), V)
    end
    return FiniteMPO(W)
end
