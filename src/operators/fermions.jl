"""
    fZ(operator::AbstractTensorMap)
    braiding operator 
"""
function fZ(operator::AbstractTensorMap)
    length(domain(operator))==2 ? vspace=domain(operator)[2] : length(codomain(operator))==2 ? vspace=codomain(operator)[1] : throw(ArgumentError("invalid creation or annihilation operator"))
    pspace = domain(operator)[1]
    iso₁ = isomorphism(storagetype(operator), vspace, vspace)
    iso₂ = isomorphism(storagetype(operator), pspace, pspace)
    @planar Z[-1 -2; -3 -4] := iso₁[-1; 1] * iso₂[-2; 2] * τ[1 2; 3 4] * iso₂[3; -3] * iso₁[4; -4]
    return Z
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
    e⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
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
function e_min(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; kwargs...)
    return e_min(ComplexF64, particle_symmetry, spin_symmetry; kwargs...)
end
function e_min(elt::Type{<:Number}, particle_symmetry::Type{U1Irrep}, spin_symmetry::Type{U1Irrep}; side=:L, spin=:up, filling::NTuple{2, Integer}=(1,1))
    if side === :L
        E = e_plus(elt, particle_symmetry, spin_symmetry; side=:L, spin=spin, filling=filling)'
        F = isomorphism(storagetype(E), flip(space(E, 2)), space(E, 2))
        @planar e⁻[-1; -2 -3] := E[-1 1; -2] * F[-3; 1]
    elseif side === :R
        e⁻ = permute(e_plus(elt, particle_symmetry, spin_symmetry; side=:L, spin=spin, filling=filling)',
                    ((2, 1), (3,)))
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return e⁻
end

"""
    number(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × U(1) × U(1) particle number operator
"""
function number(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    I = FermionParity ⊠ U1Irrep ⊠ U1Irrep
    P, Q = filling
    pspace = Vect[I]((0,0,-P)=>1, (0,0,2*Q-P)=>1, (1,1,Q-P)=>1, (1,-1,Q-P)=>1)
    n = TensorMap(zeros, elt, pspace ← pspace)
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
    onsite = TensorMap(zeros, elt, pspace ← pspace)
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


#===========================================================================================
    spin 1/2 fermions
    fℤ₂ × SU(2) × U(1) fermions
===========================================================================================#
"""
    e_plus(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; side=:L, filling=filling)
    fℤ₂ × SU(2) × U(1) electron creation operator
"""
function e_plus(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; side=:L, filling=filling)
    I = FermionParity ⊠ SU2Irrep ⊠ U1Irrep
    P, Q = filling
    pspace = Vect[I]((0,0,-P)=>1, (1,1//2,Q-P)=>1, (0,0,2*Q-P)=>1)
    vspace = Vect[I]((1,1//2,Q)=>1)
    if side == :L
        e⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        block(e⁺, I(0,0,2*Q-P)) .= sqrt(2)
        block(e⁺, I(1,1//2,Q-P)) .= 1
    elseif side == :R
        E = e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
        F = isomorphism(storagetype(E), vspace, flip(vspace))
        @planar e⁺[-1 -2; -3] := E[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
    end
    return e⁺
end
"""
    e_min(elt::Type{<:Number}, particle_symmetry::Type{SU2Irrep}, spin_symmetry::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) electron annihilation operator
"""
function e_min(elt::Type{<:Number}, particle_symmetry::Type{SU2Irrep}, spin_symmetry::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    if side === :L
        E = e_plus(elt, particle_symmetry, spin_symmetry; side=:L, filling=filling)'
        F = isomorphism(storagetype(E), flip(space(E, 2)), space(E, 2))
        @planar e⁻[-1; -2 -3] := E[-1 1; -2] * F[-3; 1]
    elseif side === :R
        e⁻ = permute(e_plus(elt, particle_symmetry, spin_symmetry; side=:L, filling=filling)',
                    ((2, 1), (3,)))
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return e⁻
end

"""
    number(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    fℤ₂ × SU(2) × U(1) particle number operator
"""
function number(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; filling::NTuple{2, Integer}=(1,1))
    I = FermionParity ⊠ SU2Irrep ⊠ U1Irrep
    P, Q = filling
    pspace = Vect[I]((0,0,-P)=>1, (1,1//2,Q-P)=>1, (0,0,2*Q-P)=>1)
    n = TensorMap(zeros, elt, pspace ← pspace)
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
    onsite = TensorMap(zeros, elt, pspace ← pspace)
    block(onsite, I((0,0,2*Q-P))) .= 1
    return onsite
end

"""
    S_plus(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; kwargs...)
    S_plus(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; side=:L)
    fℤ₂ × SU(2) × U(1) spin operator (-S⁺/√2, Sᶻ, S⁻/√2)
"""
function S_plus end
function S_plus(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; kwargs...)
    return S_plus(ComplexF64, particle_symmetry, spin_symmetry; kwargs...)
end
function S_plus(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; side=:L, filling::NTuple{2, Integer}=(1,1))
    I = FermionParity ⊠ SU2Irrep ⊠ U1Irrep
    P, Q = filling
    pspace =Vect[I]((0,0,-P)=>1, (1,1//2,Q-P)=>1, (0,0,2*Q-P)=>1)
    vspace = Vect[I]((0,1,0)=>1)
    if side == :L
        Sₑ⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
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
function S_min(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; kwargs...)
    return S_min(ComplexF64, particle_symmetry, spin_symmetry; kwargs...)
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
    S2 = TensorMap(zeros, elt, pspace ← pspace)
    block(S2, I((1,1//2,Q-P))) .= 3/4
    return S2
end


#==========================================================================================
    spin 1/2 fermions (realized by hard-core bosons and Jordan-Wigner transformation)
    include U(1) × U(1) fermions, 
    especially, for U(1)×U(1) fermions, there are
            c^†_↑  L: a^†_↑ ⊗ Z_↓         R: a^†_↑
            c_↑    L: a_↑ ⊗ Z_↓           R: a_↑
            c^†_↓  L: a^†_↓               R: Z_↑ ⊗ a^†_↓ == a^†_↓
            c_↓    L: a_↓                 R: Z_↑ ⊗ a_↓
==========================================================================================#

# function b_plus(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L, spin=:up)
#     I = U1Irrep ⊠ U1Irrep
#     pspace, vuspace, vdspace = Vect[(I)]((-1,0)=>1, (1,0)=>1,  (0,1)=>1, (0,-1)=>1), Vect[I]((1,1)=>1), Vect[I]((-1,1)=>1)
#     if spin == :up
#         if side == :L
#             b⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vuspace)
#             block(b⁺, I(1,0)) .= 1
#             block(b⁺, I(0,1)) .= -1
#         elseif side == :R
#             b⁺ = TensorMap(ones, elt, flip(vuspace)' ⊗ pspace ← pspace)
#         end
#     elseif spin == :down
#         if side == :L
#             b⁺ = TensorMap(ones, elt, pspace ← pspace ⊗ vdspace)
#         elseif side == :R
#             b⁺ = TensorMap(zeros, elt, flip(vdspace)' ⊗ pspace ← pspace)
#             block(b⁺, I(1,0)) .= -1
#             block(b⁺, I(0,-1)) .= 1
#         end
#     end
#     return b⁺
# end
# b_plus(reps, side, spin) = b_plus(reps[1], reps[2], reps[3]; side=side, spin=spin)

# function b_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L, spin=:up)
#     I = U1Irrep ⊠ U1Irrep
#     pspace, vuspace, vdspace = Vect[(I)]((-1,0)=>1, (1,0)=>1, (0,1)=>1, (0,-1)=>1), Vect[I]((1,1)=>1), Vect[I]((-1,1)=>1)
#     if spin == :up
#         if side == :L
#             b⁻ = TensorMap(zeros, elt, pspace ← pspace ⊗ flip(vuspace)')
#             block(b⁻, I(-1,0)) .= -1
#             block(b⁻, I(0,-1)) .= 1
#         elseif side == :R
#             b⁻ = TensorMap(ones, elt, vuspace ⊗ pspace ← pspace)
#         end
#     elseif spin == :down
#         if side == :L
#             b⁻ = TensorMap(ones, elt, pspace ← pspace ⊗ flip(vdspace)')
#         elseif side == :R
#             b⁻ = TensorMap(zeros, elt, vdspace ⊗ pspace ← pspace)
#             block(b⁻, I(-1,0)) .= 1
#             block(b⁻, I(0,1)) .= -1
#         end
#     end
#     return b⁻
# end
# b_min(reps, side, spin) = b_min(reps[1], reps[2], reps[3]; side=side, spin=spin)



# function Sb_plus(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L)
#     bp = b_plus(elt, U1Irrep, U1Irrep; side=side, spin=:up)
#     bm = b_min(elt, U1Irrep, U1Irrep; side=side, spin=:down)
#     if side == :L
#         iso = isomorphism(storagetype(bp), fuse(domain(bm)[2],domain(bp)[2]), domain(bm)[2]*domain(bp)[2])
#         @planar S⁺[-1; -2 -3] := bm[1; -2 2] * bp[-1; 1 3] * conj(iso[-3; 2 3])
#     elseif side == :R
#         iso = isomorphism(storagetype(bp), fuse(codomain(bm)[1],codomain(bp)[1]), codomain(bm)[1]*codomain(bp)[1])
#         @planar S⁺[-1 -2; -3] := iso[-1; 2 3] * bp[3 -2; 1] * bm[2 1; -3]
#     end
#     return S⁺
# end

# function Sb_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L)
#     bp = b_plus(elt, U1Irrep, U1Irrep; side=side, spin=:down)
#     bm = b_min(elt, U1Irrep, U1Irrep; side=side, spin=:up)
#     if side == :L
#         iso = isomorphism(storagetype(bp), fuse(domain(bm)[2],domain(bp)[2]), domain(bm)[2]*domain(bp)[2])
#         @planar S⁻[-1; -2 -3] := bm[1; -2 2] * bp[-1; 1 3] * conj(iso[-3; 2 3])
#     elseif side == :R
#         iso = isomorphism(storagetype(bp), fuse(codomain(bm)[1],codomain(bp)[1]), codomain(bm)[1]*codomain(bp)[1])
#         @planar S⁻[-1 -2; -3] := iso[-1; 2 3] * bp[3 -2; 1] * bm[2 1; -3]
#     end
#     return S⁻
# end

# function Sb_z(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep})
#     bpu = b_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:up)
#     bmu = b_min(elt, U1Irrep, U1Irrep; side=:L, spin=:up)
#     bpd = b_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:down)
#     bmd = b_min(elt, U1Irrep, U1Irrep; side=:L, spin=:down)
#     isou = isomorphism(storagetype(bpu), domain(bpu)[2], flip(domain(bpu)[2]))
#     isod = isomorphism(storagetype(bpd), domain(bpd)[2], flip(domain(bpd)[2]))
#     @planar Szu[-1; -2] := bpu[-1; 1 2] * isou[2; 3] * bmu[1; -2 3]
#     @planar Szd[-1; -2] := bpd[-1; 1 2] * isod[2; 3] * bmd[1; -2 3]
#     return (Szu - Szd)/2
# end

