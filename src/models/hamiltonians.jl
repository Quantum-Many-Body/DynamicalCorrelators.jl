#pre-defined Hamiltonians
"""
    hubbard(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, lattice=InfiniteChain(1); t=1.0, U=1.0, μ=0.0, filling=(1,1))
    fℤ₂ × U(1) × U(1) single-band Hubbard model
"""
function hubbard(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, lattice::MLattice; t=1.0, U=1.0, μ=0.0, filling=(1,1))
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
function hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, lattice::MLattice; t=1.0, U=1.0, μ=0.0, filling=(1,1))
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
    extended_hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, lattice=InfiniteChain(1); t=1.0, U=1.0, μ=0.0, filling=(1,1))
    fℤ₂ × SU(2) × U(1) single-band Hubbard model
"""
function extended_hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, lattice::MLattice; t=1.0, U=1.0, V=0.5, μ=0.0, filling=(1,1))
    hoppings = hopping(elt, SU2Irrep, U1Irrep; filling=filling)
    interaction = onsiteCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
    interaction2 = neiborCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
    numbers = number(elt, SU2Irrep, U1Irrep; filling=filling)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return -t*hoppings{i,j} + V*interaction2{i,j}
        end +
        sum(vertices(lattice)) do i
            return U*interaction{i} - μ*numbers{i}
        end
    end
end

"""
    hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, 
                    lattice::CustomLattice; kwargs...)
    fℤ₂ × SU(2) × U(1)  Hubbard model
"""
function hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, 
                        lattice::CustomLattice;
                        t = 1.0, 
                        t2 = 0.0, 
                        th = 1.0, 
                        th2 = 0.0, 
                        U = 6.0, 
                        μ = 0.0, 
                        pinning = nothing,
                        filling=(1,1))
    hop = hopping(elt, SU2Irrep, U1Irrep; filling=filling)
    onc = onsiteCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
    num = number(elt, SU2Irrep, U1Irrep; filling=filling)
    terms = []
    if length(lattice.lattice[1]) == 3
        tb = twosite_bonds(lattice, 1, 1; intralayer=true, neighbors=Neighbors(1=>Neighbors(lattice.lattice, 2)[1]))
        tb2 = twosite_bonds(lattice, 1, 1; intralayer=true, neighbors=Neighbors(2=>Neighbors(lattice.lattice, 2)[2]))
        for i in eachindex(tb)
            push!(terms, tb[i]=>-t*hop)
        end
        for i in eachindex(tb2)
            push!(terms, tb2[i]=>-t2*hop)
        end
        tf = twosite_bonds(lattice, 1, 1; intralayer=false, neighbors=Neighbors(1=>Neighbors(lattice.lattice, 2)[1]))
        tf2 = twosite_bonds(lattice, 1, 1; intralayer=false, neighbors=Neighbors(2=>Neighbors(lattice.lattice, 2)[2]))
        for i in eachindex(tf)
            push!(terms, tf[i]=>-th*hop)
        end
        for i in eachindex(tf2)
            push!(terms, tf2[i]=>-th2*hop)
        end
    elseif length(lattice.lattice[1]) == 2
        tb = twosite_bonds(lattice, 1, 1; neighbors=Neighbors(1=>Neighbors(lattice.lattice, 2)[1]))
        tb2 = twosite_bonds(lattice, 1, 1; neighbors=Neighbors(2=>Neighbors(lattice.lattice, 2)[2]))
        for i in eachindex(tb)
            push!(terms, tb[i]=>-t*hop)
        end
        for i in eachindex(tb2)
            push!(terms, tb2[i]=>-t2*hop)
        end
    end
    ob = onesite_bonds(lattice, 1)
    for i in eachindex(ob)
        push!(terms, ob[i]=>-μ*num) 
        push!(terms, ob[i]=>U*onc)
    end
    if !isnothing(pinning)
        for i in eachindex(pinning[1])
            push!(terms, pinning[1][i] => pinning[2][i]*num)
        end
    end
    I = ProductSector{Tuple{FermionParity, SU2Irrep, U1Irrep}}
    P, Q = filling
    pspace = Vect[I]((0,0,-P) => 1, (0,0,2*Q-P) => 1, (1,1//2,Q-P) => 1)
    return FiniteMPOHamiltonian(fill(pspace, sum(length,lattice.indices)), terms...)
end
"""
    hubbard_bilayer_2band(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, 
                    lattice=BilayerSquare(2, 2; norbit=2); kwargs...)
    fℤ₂ × SU(2) × U(1) two-band bilayer square lattice Hubbard model
"""
function hubbard_bilayer_2band(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, 
                        lattice=BilayerSquare(2, 2; norbit=2); 
                        pinning = nothing,
                        tzz10 = -0.110,
                        tzz20 = -0.017,
                        txx10 = -0.483,
                        txx20 = 0.069,
                        tzz1z = -0.635,
                        tzz2z = 0.0,
                        txx1z = 0.005,
                        txx2z = 0.0,
                        txz10 = 0.239,
                        txz2z = 0.034,
                        muz = 10.409,
                        mux = 10.776, 
                        U = 3.7,
                        Up = 2.5,
                        J = -0.6*2,
                        J2 = 0.6,
                        UpJ2 = Up - J2/2,
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
        if U !== 0
            push!(terms, a[i]=>U*onc)
        end
    end
    b = onesite_bonds(lattice, 2)
    for i in eachindex(b)
        push!(terms, b[i]=>mux*num)
        if U !== 0
            push!(terms, b[i]=>U*onc)
        end
    end
    ab = onesite_bonds(lattice, 1, 2)
    for i in eachindex(ab)
        if UpJ2 !== 0
            push!(terms, ab[i]=>UpJ2*nbc)
        end
        if J !== 0
            push!(terms, ab[i]=>J*sf)
        end
        if J2 !== 0
            push!(terms, ab[i]=>J2*ph)
        end
    end
    if !isnothing(pinning)
        for i in eachindex(pinning[1])
            push!(terms, pinning[1][i] => pinning[2][i]*num)
        end
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

"""
    heisenberg_model(elt::Type{<:Number}, ::Type{SU2Irrep}, lattice=FiniteChain(1); J=1.0)
"""
function heisenberg_model(elt::Type{<:Number}, ::Type{SU2Irrep}, lattice=FiniteChain(1); J=1.0,  spin=1//2)
    hei = heisenberg(elt, SU2Irrep; spin=spin)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return J * hei{i, j}
        end
    end
end

function JKGGp_model(L, x_indices, y_indices, z_indices; spin=1//2, J=1, K=0, G=0, Gp=0)
    Sx = S_x(ComplexF64, Trivial; spin=spin)
    Sy = S_y(ComplexF64, Trivial; spin=spin)
    Sz = S_z(ComplexF64, Trivial; spin=spin)
    S11 = contract_twosite(Sx, Sx)
    S12 = contract_twosite(Sx, Sy)
    S13 = contract_twosite(Sx, Sz)
    S21 = contract_twosite(Sy, Sx)
    S22 = contract_twosite(Sy, Sy)
    S23 = contract_twosite(Sy, Sz)
    S31 = contract_twosite(Sz, Sx)
    S32 = contract_twosite(Sz, Sy)
    S33 = contract_twosite(Sz, Sz)
    x_exchanges = (J+K)*S11 + Gp*(S12+S13+S21+S31) + J*(S22+S33) + G*(S23+S32)
    y_exchanges = (J+K)*S22 + Gp*(S12+S21+S23+S32) + J*(S11+S33) + G*(S13+S31)
    z_exchanges = (J+K)*S33 + Gp*(S13+S23+S31+S32) + J*(S11+S22) + G*(S12+S21)
    terms = []
    for i in x_indices
        push!(terms, i=>x_exchanges)
    end
    for i in y_indices
        push!(terms, i=>y_exchanges)
    end
    for i in z_indices
        push!(terms, i=>z_exchanges)
    end
    return FiniteMPOHamiltonian(fill(domain(Sz,1), L), terms...)
end