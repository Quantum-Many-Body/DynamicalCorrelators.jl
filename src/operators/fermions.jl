"""
===========================================================================================
    spin 1/2 fermions (realized by hard-core bosons and Jordan-Wigner transformation)
    include SU(2) × U(1), U(1) × U(1) and SU(2) × SU(2) fermions, 
    especially, for U(1)×U(1) fermions, there are
            c^†_↑  L: a^†_↑ ⊗ Z_↓         R: a^†_↑
            c_↑    L: a_↑ ⊗ Z_↓           R: a_↑
            c^†_↓  L: a^†_↓               R: Z_↑ ⊗ a^†_↓ == a^†_↓
            c_↓    L: a_↓                 R: Z_↑ ⊗ a_↓
===========================================================================================
"""
function b_plus(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L, spin=:up)
    I = Irrep[U₁] ⊠ Irrep[U₁]
    pspace, vuspace, vdspace = Vect[(I)]((-1,0)=>1, (1,0)=>1,  (0,1)=>1, (0,-1)=>1), Vect[I]((1,1)=>1), Vect[I]((-1,1)=>1)
    if spin == :up
        if side == :L
            b⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vuspace)
            blocks(b⁺)[I(1,0)] .= 1
            blocks(b⁺)[I(0,1)] .= -1
        elseif side == :R
            b⁺ = TensorMap(ones, elt, flip(vuspace)' ⊗ pspace ← pspace)
        end
    elseif spin == :down
        if side == :L
            b⁺ = TensorMap(ones, elt, pspace ← pspace ⊗ vdspace)
        elseif side == :R
            b⁺ = TensorMap(zeros, elt, flip(vdspace)' ⊗ pspace ← pspace)
            blocks(b⁺)[I(1,0)] .= -1
            blocks(b⁺)[I(0,-1)] .= 1
        end
    end
    return b⁺
end
b_plus(reps, side, spin) = b_plus(reps[1], reps[2], reps[3]; side=side, spin=spin)

function b_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L, spin=:up)
    I = Irrep[U₁] ⊠ Irrep[U₁]
    pspace, vuspace, vdspace = Vect[(I)]((-1,0)=>1, (1,0)=>1,  (0,1)=>1, (0,-1)=>1), Vect[I]((1,1)=>1), Vect[I]((-1,1)=>1)
    if spin == :up
        if side == :L
            b⁻ = TensorMap(zeros, elt, pspace ← pspace ⊗ flip(vuspace)')
            blocks(b⁻)[I(-1,0)] .= -1
            blocks(b⁻)[I(0,-1)] .= 1
        elseif side == :R
            b⁻ = TensorMap(ones, elt, vuspace ⊗ pspace ← pspace)
        end
    elseif spin == :down
        if side == :L
            b⁻ = TensorMap(ones, elt, pspace ← pspace ⊗ flip(vdspace)')
        elseif side == :R
            b⁻ = TensorMap(zeros, elt, vdspace ⊗ pspace ← pspace)
            blocks(b⁻)[I(-1,0)] .= 1
            blocks(b⁻)[I(0,1)] .= -1
        end
    end
    return b⁻
end
b_min(reps, side, spin) = b_min(reps[1], reps[2], reps[3]; side=side, spin=spin)

function fZ(operator::TensorMap)
    length(domain(operator))==2 ? vspace=domain(operator)[2] : length(codomain(operator))==2 ? vspace=codomain(operator)[1] : throw(ArgumentError("invalid creation or annihilation operator"))
    pspace = domain(operator)[1]
    T, elt = sectortype(pspace).parameters[1], eltype(operator)
    return fZ(T, elt, pspace, vspace)
end

function fZ(::Type{Tuple{U1Irrep, U1Irrep}}, elt, pspace, vspace)
    TensorMap(elt[-1 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 1], vspace*pspace, pspace*vspace)
end

function S_plus(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L)
    bp = b_plus(elt, U1Irrep, U1Irrep; side=side, spin=:up)
    bm = b_min(elt, U1Irrep, U1Irrep; side=side, spin=:down)
    if side == :L
        iso = isomorphism(storagetype(bp), fuse(domain(bm)[2],domain(bp)[2]), domain(bm)[2]*domain(bp)[2])
        @planar S⁺[-1; -2 -3] := bm[1; -2 2] * bp[-1; 1 3] * conj(iso[-3; 2 3])
    elseif side == :R
        iso = isomorphism(storagetype(bp), fuse(codomain(bm)[1],codomain(bp)[1]), codomain(bm)[1]*codomain(bp)[1])
        @planar S⁺[-1 -2; -3] := iso[-1; 2 3] * bp[3 -2; 1] * bm[2 1; -3]
    end
    return S⁺
end

function S_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L)
    bp = b_plus(elt, U1Irrep, U1Irrep; side=side, spin=:down)
    bm = b_min(elt, U1Irrep, U1Irrep; side=side, spin=:up)
    if side == :L
        iso = isomorphism(storagetype(bp), fuse(domain(bm)[2],domain(bp)[2]), domain(bm)[2]*domain(bp)[2])
        @planar S⁻[-1; -2 -3] := bm[1; -2 2] * bp[-1; 1 3] * conj(iso[-3; 2 3])
    elseif side == :R
        iso = isomorphism(storagetype(bp), fuse(codomain(bm)[1],codomain(bp)[1]), codomain(bm)[1]*codomain(bp)[1])
        @planar S⁻[-1 -2; -3] := iso[-1; 2 3] * bp[3 -2; 1] * bm[2 1; -3]
    end
    return S⁻
end

function S_z(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep})
    bpu = b_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:up)
    bmu = b_min(elt, U1Irrep, U1Irrep; side=:L, spin=:up)
    bpd = b_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:down)
    bmd = b_min(elt, U1Irrep, U1Irrep; side=:L, spin=:down)
    isou = isomorphism(storagetype(bpu), domain(bpu)[2], flip(domain(bpu)[2]))
    isod = isomorphism(storagetype(bpd), domain(bpd)[2], flip(domain(bpd)[2]))
    @planar Szu[-1; -2] := bpu[-1; 1 2] * isou[2; 3] * bmu[1; -2 3]
    @planar Szd[-1; -2] := bpd[-1; 1 2] * isod[2; 3] * bmd[1; -2 3]
    return (Szu - Szd)/2
end

# #===========================================================================================
#     spin 1/2 fermions
#     include SU(2) × U(1), U(1) × U(1) and SU(2) × SU(2) fermions
# ===========================================================================================#
# function e_plus end
# function e_plus(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; kwargs...)
#     return e_plus(ComplexF64, particle_symmetry, spin_symmetry; kwargs...)
# end

# function e_plus(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial, ::Type{Trivial}=Trivial; side=:L)
# pspace = Vect[fℤ₂](0 => 2, 1 => 2)
# vspace = Vect[fℤ₂](1 => 2)
#     if side == :L
#     e⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
#     blocks(e⁺)[fℤ₂(0)][2, 2:3] .= [one(elt), -one(elt)]
#     blocks(e⁺)[fℤ₂(1)][:, 1:2] .= [one(elt) zero(elt); zero(elt) one(elt)]
#     elseif side == :R
#     E = e_plus(elt, Trivial, Trivial; side=:L)
#     F = isomorphism(storagetype(E), vspace, flip(vspace))
#     @planar e⁺[-1 -2; -3] := E[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
#     else
#     throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
#     end
#     return e⁺
# end

# function e_plus(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}; side=:L)
#     I = fℤ₂ ⊠ SU2Irrep ⊠ U1Irrep
#     pspace = Vect[I]((0,0,-1)=>1, (1,1//2,0)=>1, (0,0,1)=>1)
#     vspace = Vect[I]((1,1//2, 1) => 1)
#     if side == :L
#         e⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
#         blocks(e⁺)[I(0,0,1)] .= sqrt(2)
#         blocks(e⁺)[I(1,1//2,0)] .= 1
#     elseif side == :R
#         E = e_plus(elt, SU2Irrep, U1Irrep; side=:L)
#         F = isomorphism(storagetype(E), vspace, flip(vspace))
#         @planar e⁺[-1 -2; -3] := E[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
#     end
#     return e⁺
# end

# function e_plus(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; side=:L, spin=:up)
#     I = fℤ₂ ⊠ Irrep[U₁] ⊠ Irrep[U₁]
#     pspace = Vect[(I)]((1,-1,0)=>1, (1,1,0)=>1,  (0,0,1)=>1, (0,0,-1)=>1)
#     vuspace,vdspace = Vect[I]((1,1,1) => 1), Vect[I]((1,-1,1) => 1)
#     if spin == :up
#         if side == :L
#             e⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vuspace)
#             blocks(e⁺)[I(1,1,0)] .= 1
#             blocks(e⁺)[I(0,0,1)] .= 1
#         elseif side == :R
#             E = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:up)
#             F = isomorphism(storagetype(E), vuspace, flip(vuspace))
#             @planar e⁺[-1 -2; -3] := E[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
#         end
#     elseif spin == :down
#         if side == :L
#             e⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vdspace)
#             blocks(e⁺)[I(1,-1,0)] .= 1
#             blocks(e⁺)[I(0,0,1)] .= -1
#         elseif side == :R
#             E = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:down)
#             F = isomorphism(storagetype(E), vdspace, flip(vdspace))
#             @planar e⁺[-1 -2; -3] := E[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
#         end
#     end
#     return e⁺
# end
# const e⁺ = e_plus

# function e_min end
# function e_min(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; kwargs...)
#     return e_min(ComplexF64, particle_symmetry, spin_symmetry; kwargs...)
# end
# function e_min(elt::Type{<:Number}=ComplexF64,
#                 particle_symmetry::Type{<:Sector}=Trivial,
#                 spin_symmetry::Type{<:Sector}=Trivial;
#                 side=:L, spin=spin)
#     if side === :L
#         E = e_plus(elt, particle_symmetry, spin_symmetry; side=:L, spin=spin)'
#         F = isomorphism(storagetype(E), flip(space(E, 2)), space(E, 2))
#         @planar e⁻[-1; -2 -3] := E[-1 1; -2] * F[-3; 1]
#     elseif side === :R
#         e⁻ = permute(e_plus(elt, particle_symmetry, spin_symmetry; side=:L, spin=spin)', ((2, 1), (3,)))
#     else
#         throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
#     end
#     return e⁻
# end
# const e⁻ = e_min