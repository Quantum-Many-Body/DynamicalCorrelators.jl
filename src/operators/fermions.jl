"""
    fZ(operator::AbstractTensorMap)

Return the materialized fermionic string tensor associated with `operator`.

The string is `TensorMap(BraidingTensor(pspace, vspace))`, where `pspace` is the
physical space and `vspace` is the charged virtual leg of the operator. Using a
concrete `TensorMap` fixes the braiding channel for string propagation through
MPO/MPS transfer matrices, which is important for fermionic signs.
"""
function fZ(operator::AbstractTensorMap)
    if length(domain(operator)) > length(codomain(operator))
        vspace = domain(operator)[length(domain(operator))]
        pspace = domain(operator)[1]
    elseif length(codomain(operator)) > length(domain(operator))
        vspace = codomain(operator)[1]
        pspace = domain(operator)[1]
    end
    return TensorMap(BraidingTensor(pspace, vspace))
end

#===========================================================================================
    spin 1/2 fermions
    fℤ₂ × U(1) × U(1) fermions
===========================================================================================#
"""
    e_plus(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L, spin=:up, filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × U(1) × U(1) electron creation operator
"""
function e_plus(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L, spin=:up, filling::NTuple{2, Integer}=(1,1))
    I = FermionParity ⊠ U1Irrep ⊠ U1Irrep
    P, Q = filling
    pspace = Vect[I]((0,0,-P)=>1, (0,0,2*Q-P)=>1, (1,1,Q-P)=>1, (1,-1,Q-P)=>1)
    vspace = spin == :up ? Vect[I]((1,1,Q)=>1) : spin == :down ? Vect[I]((1,-1,Q)=>1) : throw(ArgumentError("only support spin 1/2 operators"))
    e⁺ = zeros(elt, pspace ← pspace ⊗ vspace)
    if (side == :L)&&(spin == :up)
        block(e⁺, I((1,1,Q-P))) .= 1
        block(e⁺, I((0,0,2*Q-P))) .= -1
    elseif (side == :L)&&(spin == :down)
        block(e⁺, I((1,-1,Q-P))) .= 1
        block(e⁺, I((0,0,2*Q-P))) .= 1
    elseif side == :R
        E = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=spin, filling=filling)
        F = isomorphism(storagetype(E), vspace, flip(vspace))
        @planar e⁺[-1 -2; -3] := E[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
    end
    return e⁺
end

"""
    e_min(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; kwargs...)
    e_min(elt::Type{<:Number}, particle_symmetry::Type{U1Irrep}, spin_symmetry::Type{U1Irrep}; side=:L, spin=:up, filling=(1,1))
    fℤ₂ × U(1) × U(1) electron annihilation operator
"""
function e_min end
function e_min(spin_symmetry::Type{<:Sector}, particle_symmetry::Type{<:Sector}; kwargs...)
    return e_min(ComplexF64, spin_symmetry, particle_symmetry; kwargs...)
end
function e_min(elt::Type{<:Number}, spin_symmetry::Type{U1Irrep}, particle_symmetry::Type{U1Irrep}; side=:L, spin=:up, filling::NTuple{2, Integer}=(1,1))
    if side === :L
        E = e_plus(elt, spin_symmetry, particle_symmetry; side=:L, spin=spin, filling=filling)'
        F = isomorphism(storagetype(E), flip(space(E, 2)), space(E, 2))
        @planar e⁻[-1; -2 -3] := E[-1 1; -2] * F[-3; 1]
    elseif side === :R
        e⁻ = permute(e_plus(elt, spin_symmetry, particle_symmetry; side=:L, spin=spin, filling=filling)',
                    ((2, 1), (3,)))
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return e⁻
end

"""
    hopping(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × U(1) × U(1) hopping term
"""
function hopping(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    c⁺ul = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
    cur = e_min(elt, U1Irrep, U1Irrep; side=:R, spin=:up, filling=filling)
    c⁺dl = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
    cdr = e_min(elt, U1Irrep, U1Irrep; side=:R, spin=:down, filling=filling)
    cul = e_min(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
    c⁺ur = e_plus(elt, U1Irrep, U1Irrep; side=:R, spin=:up, filling=filling)
    cdl = e_min(elt, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
    c⁺dr = e_plus(elt, U1Irrep, U1Irrep; side=:R, spin=:down,filling=filling)
    return contract_twosite(c⁺ul,cur) + contract_twosite(c⁺dl,cdr) + contract_twosite(cul, c⁺ur) + contract_twosite(cdl, c⁺dr)
end

"""
    σz_hopping(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × U(1) × U(1) σz hopping term
"""
function σz_hopping(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    c⁺ul = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
    cur = e_min(elt, U1Irrep, U1Irrep; side=:R, spin=:up, filling=filling)
    c⁺dl = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
    cdr = e_min(elt, U1Irrep, U1Irrep; side=:R, spin=:down, filling=filling)
    cul = e_min(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
    c⁺ur = e_plus(elt, U1Irrep, U1Irrep; side=:R, spin=:up, filling=filling)
    cdl = e_min(elt, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
    c⁺dr = e_plus(elt, U1Irrep, U1Irrep; side=:R, spin=:down,filling=filling)
    return contract_twosite(c⁺ul,cur) - contract_twosite(c⁺dl,cdr) + contract_twosite(cul, c⁺ur) - contract_twosite(cdl, c⁺dr)
end

"""
    number(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × U(1) × U(1) particle number operator
"""
function number(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    I = FermionParity ⊠ U1Irrep ⊠ U1Irrep
    P, Q = filling
    pspace = Vect[I]((0,0,-P)=>1, (0,0,2*Q-P)=>1, (1,1,Q-P)=>1, (1,-1,Q-P)=>1)
    n = zeros(elt, pspace ← pspace)
    block(n, I((0,0,2*Q-P))) .= 2
    block(n, I((1,1,Q-P))) .= 1
    block(n, I((1,-1,Q-P))) .= 1
    return n
end

"""
    onsiteCoulomb(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × U(1) × U(1) onsite Coulomb interaction operator
"""
function onsiteCoulomb(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    I = FermionParity ⊠ U1Irrep ⊠ U1Irrep
    P, Q = filling
    pspace = Vect[I]((0,0,-P)=>1, (0,0,2*Q-P)=>1, (1,1,Q-P)=>1, (1,-1,Q-P)=>1)
    onsite = zeros(elt, pspace ← pspace)
    block(onsite, I((0,0,2*Q-P))) .= 1
    return onsite
end

"""
    S_plus(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × U(1) × U(1) S⁺ operator
"""
function S_plus(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L, filling=(1,1))
    cp = e_plus(elt, U1Irrep, U1Irrep; side=side, spin=:up, filling=filling)
    cm = e_min(elt, U1Irrep, U1Irrep; side=side, spin=:down, filling=filling)
    if side == :L
        iso = isomorphism(storagetype(cp), fuse(domain(cm)[2],domain(cp)[2]), domain(cm)[2]*domain(cp)[2])
        @planar S⁺[-1; -2 -3] := cm[1; -2 2] * cp[-1; 1 3] * conj(iso[-3; 2 3])
    elseif side == :R
        iso = isomorphism(storagetype(cp), fuse(codomain(cm)[1],codomain(cp)[1]), codomain(cm)[1]*codomain(cp)[1])
        @planar S⁺[-1 -2; -3] := iso[-1; 2 3] * cp[3 -2; 1] * cm[2 1; -3]
    end
    return S⁺
end

"""
    S_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × U(1) × U(1) S⁻ operator
"""
function S_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    cp = e_plus(elt, U1Irrep, U1Irrep; side=side, spin=:down, filling=filling)
    cm = e_min(elt, U1Irrep, U1Irrep; side=side, spin=:up, filling=filling)
    if side == :L
        iso = isomorphism(storagetype(cp), fuse(domain(cm)[2],domain(cp)[2]), domain(cm)[2]*domain(cp)[2])
        @planar S⁻[-1; -2 -3] := cm[1; -2 2] * cp[-1; 1 3] * conj(iso[-3; 2 3])
    elseif side == :R
        iso = isomorphism(storagetype(cp), fuse(codomain(cm)[1],codomain(cp)[1]), codomain(cm)[1]*codomain(cp)[1])
        @planar S⁻[-1 -2; -3] := iso[-1; 2 3] * cp[3 -2; 1] * cm[2 1; -3]
    end
    return S⁻
end

"""
    S_z(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × U(1) × U(1) Sᶻ operator
"""
function S_z(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    cpu = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
    cmu = e_min(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
    cpd = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
    cmd = e_min(elt, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
    isou = isomorphism(storagetype(cpu), domain(cpu)[2], flip(domain(cpu)[2]))
    isod = isomorphism(storagetype(cpd), domain(cpd)[2], flip(domain(cpd)[2]))
    @planar Szu[-1; -2] := cpu[-1; 1 2] * isou[2; 3] * cmu[1; -2 3]
    @planar Szd[-1; -2] := cpd[-1; 1 2] * isod[2; 3] * cmd[1; -2 3]
    return (Szu - Szd)/2
end

"""
    heisenberg(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × U(1) × U(1) Heisenberg terms 
"""
function heisenberg(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    S⁺S⁻ = contract_twosite(S_plus(elt, U1Irrep, U1Irrep; side=:L, filling), S_min(elt, U1Irrep, U1Irrep; side=:R, filling))
    S⁻S⁺ = contract_twosite(S_min(elt, U1Irrep, U1Irrep; side=:L, filling), S_plus(elt, U1Irrep, U1Irrep; side=:R, filling))
    SzSz = contract_twosite(S_z(elt, U1Irrep, U1Irrep; filling), S_z(elt, U1Irrep, U1Irrep; filling))
    SS = (S⁺S⁻ + S⁻S⁺)/2 + SzSz
    return SS
end

"""
    neiborCoulomb(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, interspin::Bool; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × U(1) × U(1) n↑n↓ terms between i and j sites
"""
function neiborCoulomb(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, interspin::Bool; filling::NTuple{2, Integer}=(1,1))
    epu = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
    epd = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
    emu = e_min(elt, U1Irrep, U1Irrep; side=:R, spin=:up, filling=filling)
    emd = e_min(elt, U1Irrep, U1Irrep; side=:R, spin=:down, filling=filling)
    @planar nu[-1; -2] := epu[-1; 1 2] * τ[1 2; 3 4] * emu[3 4; -2]
    @planar nd[-1; -2] := epd[-1; 1 2] * τ[1 2; 3 4] * emd[3 4; -2]
    if interspin
        @planar Up₁[-1 -2; -3 -4] := nu[-1; -3] * nd[-2; -4]
        @planar Up₂[-1 -2; -3 -4] := nd[-1; -3] * nu[-2; -4]
        return Up₁ + Up₂
    else
        @planar UpmJ₁[-1 -2; -3 -4] := nu[-1; -3] * nu[-2; -4]
        @planar UpmJ₂[-1 -2; -3 -4] := nd[-1; -3] * nd[-2; -4]
        return UpmJ₁ + UpmJ₂
    end
end

"""
    spinflip(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × U(1) × U(1) spinflip terms
"""
function spinflip(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    @planar J₁[-1 -2; -3 -4] := S_plus(elt, U1Irrep, U1Irrep; side=:L, filling=filling)[-1; -3 1] * S_min(elt, U1Irrep, U1Irrep; side=:R, filling=filling)[1 -2; -4]
    @planar J₂[-1 -2; -3 -4] := S_min(elt, U1Irrep, U1Irrep; side=:L, filling=filling)[-1; -3 1] * S_plus(elt, U1Irrep, U1Irrep; side=:R, filling=filling)[1 -2; -4]
    return J₁ + J₂
end

"""
    pairhopping(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × U(1) × U(1) pairhopping terms
"""
function pairhopping(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    epu = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
    epd = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
    emu = e_min(elt, U1Irrep, U1Irrep; side=:R, spin=:up, filling=filling)
    emd = e_min(elt, U1Irrep, U1Irrep; side=:R, spin=:down, filling=filling)
    @planar Puudd1[-1 -2; -3 -4] := contract_twosite(epu, emu)[-1 -2; 1 2] * contract_twosite(epd,emd)[1 2; -3 -4]
    @planar Pdduu1[-1 -2; -3 -4] := contract_twosite(epd, emd)[-1 -2; 1 2] * contract_twosite(epu,emu)[1 2; -3 -4]
    epu = e_plus(elt, U1Irrep, U1Irrep; side=:R, spin=:up, filling=filling)
    epd = e_plus(elt, U1Irrep, U1Irrep; side=:R, spin=:down, filling=filling)
    emu = e_min(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
    emd = e_min(elt, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
    @planar Puudd2[-1 -2; -3 -4] := contract_twosite(emu, epu)[-1 -2; 1 2] * contract_twosite(emd,epd)[1 2; -3 -4]
    @planar Pdduu2[-1 -2; -3 -4] := contract_twosite(emd, epd)[-1 -2; 1 2] * contract_twosite(emu,epu)[1 2; -3 -4]
    return (Puudd1 + Pdduu1 + Puudd2 + Pdduu2)/2
end

#===========================================================================================
    spin 1/2 fermions
    fℤ₂ × SU(2) × U(1) fermions
===========================================================================================#
"""
    e_plus(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) electron creation operator
"""
function e_plus(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    I = FermionParity ⊠ SU2Irrep ⊠ U1Irrep
    P, Q = filling
    pspace = Vect[I]((0,0,-P)=>1, (1,1//2,Q-P)=>1, (0,0,2*Q-P)=>1)
    vspace = Vect[I]((1,1//2,Q)=>1)
    if side == :L
        e⁺ = zeros(elt, pspace ← pspace ⊗ vspace)
        block(e⁺, I(0,0,2*Q-P)) .= sqrt(2)
        block(e⁺, I(1,1//2,Q-P)) .= 1
    elseif side == :R
        E = e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
        F = isomorphism(storagetype(E), vspace, flip(vspace))
        @planar e⁺[-1 -2; -3] := E[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return e⁺
end
"""
    e_min(elt::Type{<:Number}, spin_symmetry::Type{SU2Irrep}, particle_symmetry::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) electron annihilation operator
"""
function e_min(elt::Type{<:Number}, spin_symmetry::Type{SU2Irrep}, particle_symmetry::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    if side === :L
        E = e_plus(elt, spin_symmetry, particle_symmetry; side=:L, filling=filling)'
        F = isomorphism(storagetype(E), flip(space(E, 2)), space(E, 2))
        @planar e⁻[-1; -2 -3] := E[-1 1; -2] * F[-3; 1]
    elseif side === :R
        e⁻ = permute(e_plus(elt, spin_symmetry, particle_symmetry; side=:L, filling=filling)',
                    ((2, 1), (3,)))
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return e⁻
end

"""
    hopping(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) hopping term
"""
function hopping(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    c⁺l = e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
    cr = e_min(elt, SU2Irrep, U1Irrep; side=:R, filling=filling)
    cl = e_min(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
    c⁺r = e_plus(elt, SU2Irrep, U1Irrep; side=:R, filling=filling)
    return contract_twosite(c⁺l,cr) + contract_twosite(cl, c⁺r)
end

function cdagc(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    c⁺l = e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
    cr = e_min(elt, SU2Irrep, U1Irrep; side=:R, filling=filling)
    return contract_twosite(c⁺l,cr)
end

function ccdag(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    cl = e_min(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
    c⁺r = e_plus(elt, SU2Irrep, U1Irrep; side=:R, filling=filling)
    return contract_twosite(cl, c⁺r)
end

"""
    number(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) particle number operator
"""
function number(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    I = FermionParity ⊠ SU2Irrep ⊠ U1Irrep
    P, Q = filling
    pspace = Vect[I]((0,0,-P)=>1, (1,1//2,Q-P)=>1, (0,0,2*Q-P)=>1)
    n = zeros(elt, pspace ← pspace)
    block(n, I((0,0,2*Q-P))) .= 2
    block(n, I((1,1//2,Q-P))) .= 1
    return n
end

"""
    onsiteCoulomb(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) onsite Coulomb interaction operator
"""
function onsiteCoulomb(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    I = FermionParity ⊠ SU2Irrep ⊠ U1Irrep
    P, Q = filling
    pspace = Vect[I]((0,0,-P)=>1, (1,1//2,Q-P)=>1, (0,0,2*Q-P)=>1)
    onsite = zeros(elt, pspace ← pspace)
    block(onsite, I((0,0,2*Q-P))) .= 1
    return onsite
end

"""
    S_plus(spin_symmetry::Type{<:Sector}, particle_symmetry::Type{<:Sector}; kwargs...)
    S_plus(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; side=:L)
    fℤ₂ × SU(2) × U(1) spin operator (-S⁺/√2, Sᶻ, S⁻/√2)
"""
function S_plus end
function S_plus(spin_symmetry::Type{<:Sector}, particle_symmetry::Type{<:Sector}; kwargs...)
    return S_plus(ComplexF64, spin_symmetry, particle_symmetry; kwargs...)
end
function S_plus(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    I = FermionParity ⊠ SU2Irrep ⊠ U1Irrep
    P, Q = filling
    pspace =Vect[I]((0,0,-P)=>1, (1,1//2,Q-P)=>1, (0,0,2*Q-P)=>1)
    vspace = Vect[I]((0,1,0)=>1)
    if side == :L
        Sₑ⁺ = zeros(elt, pspace ← pspace ⊗ vspace)
        block(Sₑ⁺, I(1,1//2,Q-P)) .= sqrt(3)/2
    elseif side == :R
        S = S_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
        F = isomorphism(storagetype(S), vspace, flip(vspace))
        @planar Sₑ⁺[-1 -2; -3] := S[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return Sₑ⁺
end

"""
    S_min(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; kwargs...)
    S_min(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; side=:L)
    fℤ₂ × SU(2) × U(1) spin operator (-S⁻/√2, Sᶻ, S⁺/√2)
"""
function S_min end
function S_min(spin_symmetry::Type{<:Sector}, particle_symmetry::Type{<:Sector}; kwargs...)
    return S_min(ComplexF64, spin_symmetry, particle_symmetry; kwargs...)
end
function S_min(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    if side === :L
        S = S_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)'
        F = isomorphism(storagetype(S), flip(space(S, 2)), space(S, 2))
        @planar Sₑ⁻[-1; -2 -3] := S[-1 1; -2] * F[-3; 1]
    elseif side === :R
        Sₑ⁻ = permute(S_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)', ((2, 1), (3,)))
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return Sₑ⁻
end

"""
    S_square(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep})
"""
function S_square(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    I = FermionParity ⊠ SU2Irrep ⊠ U1Irrep
    P, Q = filling
    pspace =Vect[I]((0,0,-P)=>1, (1,1//2,Q-P)=>1, (0,0,2*Q-P)=>1)
    S2 = zeros(elt, pspace ← pspace)
    block(S2, I((1,1//2,Q-P))) .= 3/4
    return S2
end

"""
    neiborCoulomb(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) n↑n↓ terms between i and j sites
"""
function neiborCoulomb(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    n = number(elt, SU2Irrep, U1Irrep; filling=filling)
    @planar nn[-1 -2; -3 -4] := n[-1; -3] * n[-2; -4]
    return nn
end

"""
    heisenberg(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) heisenberg term
"""
function heisenberg(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    return contract_twosite(S_plus(elt, SU2Irrep, U1Irrep; side=:L, filling), S_min(elt, SU2Irrep, U1Irrep; side=:R, filling))
end

"""
    pairhopping(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) pairhopping terms
"""
function pairhopping(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    ep = e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
    em = e_min(elt, SU2Irrep, U1Irrep; side=:R, filling=filling)
    @planar Pij[-1 -2; -3 -4] := contract_twosite(ep, em)[-1 -2; 1 2] * contract_twosite(ep,em)[1 2; -3 -4]
    ep = e_plus(elt, SU2Irrep, U1Irrep; side=:R, filling=filling)
    em = e_min(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
    @planar Pji[-1 -2; -3 -4] := contract_twosite(em, ep)[-1 -2; 1 2] * contract_twosite(em,ep)[1 2; -3 -4]
    return (Pij + Pji)/2
end

"""
    singlet_dagger(elt::Type{<:Number}, spin_symmetry::Type{SU2Irrep}, particle_symmetry::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) singlet creation operator
"""
function singlet_dagger(elt::Type{<:Number}, spin_symmetry::Type{SU2Irrep}, particle_symmetry::Type{U1Irrep}; onsite::Bool=false, side=:L, filling::NTuple{2, Integer}=(1,1))
    if onsite == true
        if side == :L
            A = e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
            B = deepcopy(A)
            vspace = domain(A,2)
            fspace = Vect[FermionParity ⊠ SU2Irrep ⊠ U1Irrep]((0,0,2*(filling[2]))=>1)
            iso = isometry(elt, vspace⊗vspace, fspace)
            @planar slt[-1; -2 -3] := A[-1; 1 2] * B[1; -2 3] * iso[3 2; -3]
        elseif side == :R
            sd = singlet_dagger(elt, spin_symmetry, particle_symmetry; onsite=true, side=:L, filling=filling)
            vspace = domain(sd, 2)
            F = isomorphism(storagetype(sd), vspace, flip(vspace))
            @planar slt[-1 -2; -3] := sd[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
        else
            throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
        end
    else 
        if side == :L
            A = e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
            B = deepcopy(A)
            vspace = domain(A,2)
            fspace = Vect[FermionParity ⊠ SU2Irrep ⊠ U1Irrep]((0,0,2*(filling[2]))=>1)
            iso = isometry(elt, vspace⊗vspace, fspace)
            @planar slt[-1 -2; -3 -4 -5] := A[-1; -3 1] * τ[1 2; -4 3] * B[-2; 2 4] * iso[3 4; -5]
        elseif side == :R
            sd = singlet_dagger(elt, spin_symmetry, particle_symmetry; side=:L, filling=filling)
            vspace = domain(sd, 3)
            F = isomorphism(storagetype(sd), vspace, flip(vspace))
            @planar slt[-5 -1 -2; -3 -4] := sd[-1 -2; 4 2 1] * τ[2 1; 3 -4] * τ[4 3; 5 -3] * F[5; -5]
        else
            throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
        end
    end
    return slt
end

"""
    singlet(elt::Type{<:Number}, spin_symmetry::Type{SU2Irrep}, particle_symmetry::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) singlet annihilation operator
"""
function singlet(elt::Type{<:Number}, spin_symmetry::Type{SU2Irrep}, particle_symmetry::Type{U1Irrep}; onsite::Bool=false, side=:L, filling::NTuple{2, Integer}=(1,1))
    if onsite == true
        if side === :L
            sd = singlet_dagger(elt, spin_symmetry, particle_symmetry; onsite=true, side=:L, filling=filling)'
            F = isomorphism(storagetype(sd), flip(space(sd, 2)), space(sd, 2))
            @planar slt[-1; -2 -3] := sd[-1 1; -2] * F[-3; 1]
        elseif side === :R
            slt = permute(singlet_dagger(elt, spin_symmetry, particle_symmetry; onsite=true, side=:L, filling=filling)', ((2, 1), (3,)))
        else
            throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
        end
    else
        if side == :L
            sd = singlet_dagger(elt, spin_symmetry, particle_symmetry; side=:L, filling=filling)'
            F = isomorphism(storagetype(sd), flip(space(sd, 3)), space(sd, 3))
            @planar slt[-1 -2; -3 -4 -5] := sd[-1 -2 1; -3 -4] * F[-5; 1]
        elseif side == :R
            slt = permute(singlet_dagger(elt, spin_symmetry, particle_symmetry; side=:L, filling=filling)', ((3, 1, 2), (4, 5)))
        else
            throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
        end
    end
    return slt
end

"""
    triplet_dagger(elt::Type{<:Number}, spin_symmetry::Type{SU2Irrep}, particle_symmetry::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) triplet creation operator
"""
function triplet_dagger(elt::Type{<:Number}, spin_symmetry::Type{SU2Irrep}, particle_symmetry::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    if side == :L
        A = e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
        B = deepcopy(A)
        vspace = domain(A,2)
        fspace = Vect[FermionParity ⊠ SU2Irrep ⊠ U1Irrep]((0,1,2*(filling[2]))=>1)
        iso = isometry(elt, vspace⊗vspace, fspace)
        @planar slt[-1 -2; -3 -4 -5] := A[-1; -3 1] * τ[1 2; -4 3] * B[-2; 2 4] * iso[3 4; -5]
    elseif side == :R
        sd = triplet_dagger(elt, spin_symmetry, particle_symmetry; side=:L, filling=filling)
        vspace = domain(sd, 3)
        F = isomorphism(storagetype(sd), vspace, flip(vspace))
        @planar slt[-5 -1 -2; -3 -4] := sd[-1 -2; 4 2 1] * τ[2 1; 3 -4] * τ[4 3; 5 -3] * F[5; -5]
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return slt/sqrt(3)
end

"""
    triplet(elt::Type{<:Number}, spin_symmetry::Type{SU2Irrep}, particle_symmetry::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) triplet annihilation operator
"""
function triplet(elt::Type{<:Number}, spin_symmetry::Type{SU2Irrep}, particle_symmetry::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    if side == :L
        sd = triplet_dagger(elt, spin_symmetry, particle_symmetry; side=:L, filling=filling)'
        F = isomorphism(storagetype(sd), flip(space(sd, 3)), space(sd, 3))
        @planar slt[-1 -2; -3 -4 -5] := sd[-1 -2 1; -3 -4] * F[-5; 1]
    elseif side == :R
        slt = permute(triplet_dagger(elt, spin_symmetry, particle_symmetry; side=:L, filling=filling)', ((3, 1, 2), (4, 5)))
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return slt
end

#===========================================================================================
    spin 1/2 fermions
    fℤ₂ × SU(2) fermions
===========================================================================================#
"""
    e_plus(elt::Type{<:Number}, ::Type{SU2Irrep}; side=:L)
    fℤ₂ × SU(2) electron creation operator
"""
function e_plus(elt::Type{<:Number}, ::Type{SU2Irrep}; side=:L)
    vspace = Vect[(FermionParity ⊠ SU2Irrep)]((1, 1/2) => 1)
    pspace = Vect[(FermionParity ⊠ SU2Irrep)]((0, 0) => 2, (1, 1/2) => 1)
    if side == :L
        e⁺ = TensorMap(elt[0.0, sqrt(2), 1.0, 0.0], pspace ← (pspace ⊗ vspace))
    elseif side == :R
        E = e_plus(elt, SU2Irrep; side=:L)
        F = isomorphism(storagetype(E), vspace, flip(vspace))
        @planar e⁺[-1 -2; -3] := E[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return e⁺
end
"""
    e_min(elt::Type{<:Number}, ::Type{SU2Irrep}; side=:L)
    fℤ₂ × SU(2) electron annihilation operator
"""
function e_min(elt::Type{<:Number}, ::Type{SU2Irrep}; side=:L)
    if side === :L
        E = e_plus(elt, SU2Irrep; side=:L)'
        F = isomorphism(storagetype(E), flip(space(E, 2)), space(E, 2))
        @planar e⁻[-1; -2 -3] := E[-1 1; -2] * F[-3; 1]
    elseif side === :R
        e⁻ = permute(e_plus(elt, SU2Irrep; side=:L)',
                    ((2, 1), (3,)))
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return e⁻
end

function hopping(elt::Type{<:Number}, ::Type{SU2Irrep})
    return cdagc(elt, SU2Irrep) + ccdag(elt, SU2Irrep)
end

function cdagc(elt::Type{<:Number}, ::Type{SU2Irrep})
    c⁺l = e_plus(elt, SU2Irrep; side=:L)
    cr = e_min(elt, SU2Irrep; side=:R)
    return contract_twosite(c⁺l, cr)
end

function ccdag(elt::Type{<:Number}, ::Type{SU2Irrep})
    cl = e_min(elt, SU2Irrep; side=:L)
    c⁺r = e_plus(elt, SU2Irrep; side=:R)
    return contract_twosite(cl, c⁺r)
end

function number(elt::Type{<:Number}, ::Type{SU2Irrep})
    pspace = Vect[(FermionParity ⊠ SU2Irrep)]((0, 0) => 2, (1, 1/2) => 1)
    n = zeros(elt, pspace, pspace)
    block(n,(FermionParity(0) ⊠ SU2Irrep(0)))[2,2] = 2
    block(n,(FermionParity(1) ⊠ SU2Irrep(1//2)))[1,1] = 1
    return n
end

function onsiteCoulomb(elt::Type{<:Number}, ::Type{SU2Irrep})
    pspace = Vect[(FermionParity ⊠ SU2Irrep)]((0, 0) => 2, (1, 1/2) => 1)
    nn = zeros(elt, pspace, pspace)
    block(nn,(FermionParity(0) ⊠ SU2Irrep(0)))[2,2] = 1
    return nn
end

function neiborCoulomb(elt::Type{<:Number}, ::Type{SU2Irrep})
    n = number(elt, SU2Irrep)
    @planar nn[-1 -2; -3 -4] := n[-1; -3] * n[-2; -4]
    return nn
end

function S_plus(elt::Type{<:Number}, ::Type{SU2Irrep}; side=:L)
    vspace = Vect[(FermionParity ⊠ SU2Irrep)]((0, 1) => 1)
    pspace = Vect[(FermionParity ⊠ SU2Irrep)]((0, 0) => 2, (1, 1/2) => 1)
    if side == :L
        Sₑ⁺ = zeros(Float64, pspace ← pspace ⊗ vspace)
        block(Sₑ⁺, (FermionParity(1) ⊠ SU2Irrep(1/2))) .= sqrt(3)/2
    elseif side == :R
        S = S_plus(elt, SU2Irrep; side=:L)
        F = isomorphism(storagetype(S), vspace, flip(vspace))
        @planar Sₑ⁺[-1 -2; -3] := S[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return Sₑ⁺
end

function S_min(elt::Type{<:Number}, ::Type{SU2Irrep}; side=:L)
    if side === :L
        S = S_plus(elt, SU2Irrep; side=:L)'
        F = isomorphism(storagetype(S), flip(space(S, 2)), space(S, 2))
        @planar Sₑ⁻[-1; -2 -3] := S[-1 1; -2] * F[-3; 1]
    elseif side === :R
        Sₑ⁻ = permute(S_plus(elt, SU2Irrep; side=:L)', ((2, 1), (3,)))
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return Sₑ⁻
end

function heisenberg(elt::Type{<:Number}, ::Type{SU2Irrep})
    return contract_twosite(S_plus(elt, SU2Irrep; side=:L), S_min(elt, SU2Irrep; side=:R))
end

function pairhopping(elt::Type{<:Number}, ::Type{SU2Irrep})
    ep = e_plus(elt, SU2Irrep; side=:L)
    em = e_min(elt, SU2Irrep; side=:R)
    @planar Pij[-1 -2; -3 -4] := contract_twosite(ep, em)[-1 -2; 1 2] * contract_twosite(ep,em)[1 2; -3 -4]
    ep = e_plus(elt, SU2Irrep; side=:R)
    em = e_min(elt, SU2Irrep; side=:L)
    @planar Pji[-1 -2; -3 -4] := contract_twosite(em, ep)[-1 -2; 1 2] * contract_twosite(em,ep)[1 2; -3 -4]
    return (Pij + Pji)/2
end

function singlet_dagger(elt::Type{<:Number}, ::Type{SU2Irrep}; onsite::Bool=false, side=:L)
    if onsite == true
        if side == :L
            A = e_plus(elt, SU2Irrep; side=:L)
            B = deepcopy(A)
            vspace = domain(A,2)
            fspace = Vect[FermionParity ⊠ SU2Irrep]((0,0)=>1)
            iso = isometry(elt, vspace⊗vspace, fspace)
            @planar slt[-1; -2 -3] := A[-1; 1 2] * B[1; -2 3] * iso[3 2; -3]
        elseif side == :R
            sd = singlet_dagger(elt, SU2Irrep; onsite=true, side=:L)
            vspace = domain(sd, 2)
            F = isomorphism(storagetype(sd), vspace, flip(vspace))
            @planar slt[-1 -2; -3] := sd[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
        else
            throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
        end
    else 
        if side == :L
            A = e_plus(elt, SU2Irrep; side=:L)
            B = deepcopy(A)
            vspace = domain(A,2)
            fspace = Vect[FermionParity ⊠ SU2Irrep]((0,0)=>1)
            iso = isometry(elt, vspace⊗vspace, fspace)
            @planar slt[-1 -2; -3 -4 -5] := A[-1; -3 1] * τ[1 2; -4 3] * B[-2; 2 4] * iso[3 4; -5]
        elseif side == :R
            sd = singlet_dagger(elt, SU2Irrep; side=:L)
            vspace = domain(sd, 3)
            F = isomorphism(storagetype(sd), vspace, flip(vspace))
            @planar slt[-5 -1 -2; -3 -4] := sd[-1 -2; 4 2 1] * τ[2 1; 3 -4] * τ[4 3; 5 -3] * F[5; -5]
        else
            throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
        end
    end
    return slt
end

function singlet(elt::Type{<:Number}, ::Type{SU2Irrep}; onsite::Bool=false, side=:L)
    if onsite == true
        if side === :L
            sd = singlet_dagger(elt, SU2Irrep; onsite=true, side=:L)'
            F = isomorphism(storagetype(sd), flip(space(sd, 2)), space(sd, 2))
            @planar slt[-1; -2 -3] := sd[-1 1; -2] * F[-3; 1]
        elseif side === :R
            slt = permute(singlet_dagger(elt, SU2Irrep; onsite=true, side=:L)', ((2, 1), (3,)))
        else
            throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
        end
    else
        if side == :L
            sd = singlet_dagger(elt, SU2Irrep; side=:L)'
            F = isomorphism(storagetype(sd), flip(space(sd, 3)), space(sd, 3))
            @planar slt[-1 -2; -3 -4 -5] := sd[-1 -2 1; -3 -4] * F[-5; 1]
        elseif side == :R
            slt = permute(singlet_dagger(elt, SU2Irrep; side=:L)', ((3, 1, 2), (4, 5)))
        else
            throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
        end
    end
    return slt
end
