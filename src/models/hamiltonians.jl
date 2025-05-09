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

"""
    hubbard_bilayer_2band(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, 
                    lattice=BilayerSquare(2, 2; norbit=2); kwargs...)
    fℤ₂ × SU(2) × U(1) two-band bilayer square lattice Hubbard model
"""
function hubbard_bilayer_2band(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, 
                        lattice=BilayerSquare(2, 2; norbit=2); 
                        tzz10 = -0.1123,
                        tzz20 = -0.0142,
                        txx10 = -0.4897,
                        txx20 = 0.0686,
                        tzz1z = -0.6420,
                        tzz2z = 0.0257,
                        txx1z = 0.0029,
                        txx2z = 0.0006,
                        txz10 = 0.2425,
                        txz2z = 0.0370,
                        muz = 10.5124,
                        mux = 10.8716, 
                        U = 6.0,
                        Up = 3.6,
                        UpJ = 2.4,
                        J = -1.2*2,
                        J2 = 1.2,
                        UpJ2 = Up - 1.2/2,
                        filling=(1,1))
    hop = hopping(elt, SU2Irrep, U1Irrep; filling=filling)
    onc = onsiteCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
    num = number(elt, SU2Irrep, U1Irrep; filling=filling)
    nbc = neiborCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
    sf = heisenberg(elt, SU2Irrep, U1Irrep; filling=filling)
    ph = pairhopping(elt, SU2Irrep, U1Irrep; filling=filling)
    terms = []
    if tzz10 !== 0
        zz10 = twosite_bonds(lattice, 1, 1; intralayer=true, neighbors=Neighbors(1=>1))
        for i in eachindex(zz10)
            push!(terms, zz10[i]=>tzz10*hop)
        end
    end
    if tzz20 !== 0
        zz20 = twosite_bonds(lattice, 1, 1; intralayer=true, neighbors=Neighbors(2=>sqrt(2)))
        for i in eachindex(zz20)
            push!(terms, zz20[i]=>tzz20*hop)
        end
        
    end
    if txx10 !== 0
        xx10 = twosite_bonds(lattice, 2, 2; intralayer=true, neighbors=Neighbors(1=>1))
        for i in eachindex(xx10)
            push!(terms, xx10[i]=>txx10*hop)
        end
    end
    if txx20 !== 0
        xx20 = twosite_bonds(lattice, 2, 2; intralayer=true, neighbors=Neighbors(2=>sqrt(2)))
        for i in eachindex(xx20)
            push!(terms, xx20[i]=>txx20*hop) 
        end
    end
    if tzz1z !== 0
        zz1z = twosite_bonds(lattice, 1, 1; intralayer=false, neighbors=Neighbors(1=>1))
        for i in eachindex(zz1z)
            push!(terms, zz1z[i]=>tzz1z*hop)
        end
    end
    if tzz2z !== 0
        zz2z = twosite_bonds(lattice, 1, 1; intralayer=false, neighbors=Neighbors(2=>sqrt(2)))
        for i in eachindex(zz2z)
            push!(terms, zz2z[i]=>tzz2z*hop)
        end
    end
    if txx1z !== 0
        xx1z = twosite_bonds(lattice, 2, 2; intralayer=false, neighbors=Neighbors(1=>1))
        for i in eachindex(xx1z)
            push!(terms, xx1z[i]=>txx1z*hop)
        end
    end
    if txx2z !== 0
        xx2z = twosite_bonds(lattice, 2, 2; intralayer=false, neighbors=Neighbors(2=>sqrt(2)))
        for i in eachindex(xx2z)
            push!(terms, xx2z[i]=>txx2z*hop)
        end
    end
    if txz10 !== 0
        xz10 = [twosite_bonds(lattice, 1, 2; intralayer=true, neighbors=Neighbors(1=>1)); twosite_bonds(lattice, 2, 1; intralayer=true, neighbors=Neighbors(1=>1))]
        for i in eachindex(xz10)
            a, b = find_position(lattice.indices, xz10[i][1]), find_position(lattice.indices, xz10[i][2])
            if (lattice.lattice[a] - lattice.lattice[b])[2] ≈ 0
                push!(terms, xz10[i]=>txz10*hop)
            elseif (lattice.lattice[a] - lattice.lattice[b])[1] ≈ 0
                push!(terms, xz10[i]=>-txz10*hop)
            else
                throw(ArgumentError("Invalid nn xz bond"))
            end
        end
    end
    if txz2z !== 0
        xz2z = [twosite_bonds(lattice, 1, 2; intralayer=false, neighbors=Neighbors(2=>sqrt(2))); twosite_bonds(lattice, 2, 1; intralayer=false, neighbors=Neighbors(2=>sqrt(2)))]
        for i in eachindex(xz2z)
            a, b = find_position(lattice.indices, xz2z[i][1]), find_position(lattice.indices, xz2z[i][2])
            if (lattice.lattice[a] - lattice.lattice[b])[2] ≈ 0
                push!(terms, xz2z[i]=>-txz2z*hop)
            elseif (lattice.lattice[a] - lattice.lattice[b])[1] ≈ 0
                push!(terms, xz2z[i]=>txz2z*hop)
            else
                throw(ArgumentError("Invalid nnn xz bond"))
            end 
        end
    end
    a = onesite_bonds(lattice, 1)
    for i in eachindex(a)
        push!(terms, a[i]=>muz*num) 
        push!(terms, a[i]=>U*onc)
    end
    b = onesite_bonds(lattice, 2)
    for i in eachindex(b)
        push!(terms, b[i]=>mux*num)
        push!(terms, b[i]=>U*onc)
    end
    ab = onesite_bonds(lattice, 1, 2)
    for i in eachindex(ab)
        push!(terms, ab[i]=>UpJ2*nbc)
        push!(terms, ab[i]=>J*sf)
        push!(terms, ab[i]=>J2*ph)
    end
    I = ProductSector{Tuple{FermionParity, SU2Irrep, U1Irrep}}
    P, Q = filling
    pspace = Vect[I]((0,0,-P) => 1, (0,0,2*Q-P) => 1, (1,1//2,Q-P) => 1)
    return FiniteMPOHamiltonian(fill(pspace, sum(length,lattice.indices)), terms...)
end


"""
    Kitaev_hubbard(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, lattice=FiniteChain(1); t=1.0, tz=0.0, U=1.0, μ=0.0, filling=(1,1))
    fℤ₂ × U(1) × U(1) 1d-Chain Kitaev-Hubbard model without pairing terms
"""

function kitaev_hubbard(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, lattice=FiniteChain(1); t=1.0, tz=0.0, U=1.0, μ=0.0, filling=(1,1))
    hoppings = hopping(elt, U1Irrep, U1Irrep;filling=filling)
    σz_hoppings = σz_hopping(elt, U1Irrep, U1Irrep;filling=filling)
    interaction = onsiteCoulomb(elt, U1Irrep, U1Irrep; filling=filling)
    numbers = number(elt, U1Irrep, U1Irrep; filling=filling)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return (-t*hoppings{i,j} - tz*σz_hoppings{i,j})/2
        end +
        sum(vertices(lattice)) do i
            return U*interaction{i} - μ*numbers{i}
        end
    end
end