#pre-defined Hamiltonians
"""
    hubbard(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, lattice=InfiniteChain(1); t=1.0, U=1.0, μ=0.0, filling=(1,1))
    fℤ₂ × U(1) × U(1) single-band Hubbard model
"""
function hubbard(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, lattice=InfiniteChain(1); t=1.0, U=1.0, μ=0.0, filling=(1,1))
    hoppings = hopping(elt, U1Irrep, U1Irrep;filling=filling)
    interaction = onsiteCoulomb(elt, U1Irrep, U1Irrep; filling=filling)
    numbers = number(elt, U1Irrep, U1Irrep; filling=filling)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return -t*hoppings{i,j}
        end +
        sum(vertices(lattice)) do i
            return U*interaction{i} - μ*numbers{i}
        end
    end
end

"""
    hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, lattice=InfiniteChain(1); t=1.0, U=1.0, μ=0.0, filling=(1,1))
    fℤ₂ × SU(2) × U(1) single-band Hubbard model
"""
function hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, lattice=InfiniteChain(1); t=1.0, U=1.0, μ=0.0, filling=(1,1))
    hoppings = hopping(elt, SU2Irrep, U1Irrep; filling=filling)
    interaction = onsiteCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
    numbers = number(elt, SU2Irrep, U1Irrep; filling=filling)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return -t*hoppings{i,j}
        end +
        sum(vertices(lattice)) do i
            return U*interaction{i} - μ*numbers{i}
        end
    end
end

# function hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, lattice=BilayerSquare(2, 2; norbit=2); t=1.0, U=1.0, μ=0.0, filling=(1,1))
#     hopping = contract_twosite(c⁺l,cr) + contract_twosite(cl, c⁺r)
#     onc = onsiteCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
#     num = number(elt, SU2Irrep, U1Irrep; filling=filling)
#     nbc = neiborCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
#     sf = heisenberg(elt, SU2Irrep, U1Irrep; filling=filling)
#     ph = pairhopping(elt, SU2Irrep, U1Irrep; filling=filling)
#     tz10 = -0.1123
#     tz20 = -0.0142
#     tz1z = -0.6420
#     tz2z = 0.0257
#     tx10 = -0.4897
#     tx20 = 0.0686
#     tx1z = 0.0029
#     tx2z = 0.0006
#     txz10 = 0.2425
#     txz2z = 0.0370
#     muz = 10.5124 
#     mux = 10.8716 
#     U = 6.0
#     Up = 3.6
#     UpJ = 2.4
#     J = -1.2*2
#     J2 = 1.2
#     UpJ2 = Up - 1.2/2
# end