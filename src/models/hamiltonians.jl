#pre-defined Hamiltonians
"""
    hubbard(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, lattice=InfiniteChain(1); t=1.0, U=1.0, μ=0.0, filling=(1,1))
    fℤ₂ × U(1) × U(1) single-band Hubbard model
"""
function hubbard(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, lattice=InfiniteChain(1); t=1.0, U=1.0, μ=0.0, filling=(1,1))
    c⁺ul = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
    cur = e_min(elt, U1Irrep, U1Irrep; side=:R, spin=:up, filling=filling)
    c⁺dl = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
    cdr = e_min(elt, U1Irrep, U1Irrep; side=:R, spin=:down, filling=filling)
    cul = e_min(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
    c⁺ur = e_plus(elt, U1Irrep, U1Irrep; side=:R, spin=:up, filling=filling)
    cdl = e_min(elt, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
    c⁺dr = e_plus(elt, U1Irrep, U1Irrep; side=:R, spin=:down,filling=filling)
    hopping = contract_twosite(c⁺ul,cur) + contract_twosite(c⁺dl,cdr) + contract_twosite(cul, c⁺ur) + contract_twosite(cdl, c⁺dr)
    interaction = onsiteCoulomb(elt, U1Irrep, U1Irrep; filling=filling)
    numbers = number(elt, U1Irrep, U1Irrep; filling=filling)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return -t*hopping{i,j}
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
    c⁺l = e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
    cr = e_min(elt, SU2Irrep, U1Irrep; side=:R, filling=filling)
    cl = e_min(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
    c⁺r = e_plus(elt, SU2Irrep, U1Irrep; side=:R, filling=filling)
    hopping = contract_twosite(c⁺l,cr) + contract_twosite(cl, c⁺r)
    interaction = onsiteCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
    numbers = number(elt, SU2Irrep, U1Irrep; filling=filling)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return -t*hopping{i,j}
        end +
        sum(vertices(lattice)) do i
            return U*interaction{i} - μ*numbers{i}
        end
    end
end