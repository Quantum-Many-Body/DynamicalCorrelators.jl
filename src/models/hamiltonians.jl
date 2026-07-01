#pre-defined Hamiltonians
"""
    hubbard(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, lattice=InfiniteChain(1); t=1.0, U=1.0, mu=0.0, filling=(1,1))
    fℤ₂ × U(1) × U(1) single-band Hubbard model
"""
function hubbard(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, lattice::MLattice; t=1.0, U=1.0, mu=0.0, filling=(1,1))
    hoppings = hopping(elt, U1Irrep, U1Irrep;filling=filling)
    interaction = onsiteCoulomb(elt, U1Irrep, U1Irrep; filling=filling)
    numbers = number(elt, U1Irrep, U1Irrep; filling=filling)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return -t*hoppings{i,j}
        end +
        sum(vertices(lattice)) do i
            return U*interaction{i} - mu*numbers{i}
        end
    end
end

"""
    hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, lattice=InfiniteChain(1); t=1.0, U=1.0, mu=0.0, filling=(1,1))
    fℤ₂ × SU(2) × U(1) single-band Hubbard model
"""
function hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, lattice::MLattice; t=1.0, U=1.0, mu=0.0, filling=(1,1))
    hoppings = hopping(elt, SU2Irrep, U1Irrep; filling=filling)
    interaction = onsiteCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
    numbers = number(elt, SU2Irrep, U1Irrep; filling=filling)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return -t*hoppings{i,j}
        end +
        sum(vertices(lattice)) do i
            return U*interaction{i} - mu*numbers{i}
        end
    end
end

"""
    hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, lattice=InfiniteChain(1); t=1.0, U=1.0, mu=0.0)
    fℤ₂ × SU(2) single-band Hubbard model
"""
function hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, lattice::MLattice; t=1.0, U=1.0, mu=0.0)
    hoppings = hopping(elt, SU2Irrep)
    interaction = onsiteCoulomb(elt, SU2Irrep)
    numbers = number(elt, SU2Irrep)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return -t*hoppings{i,j}
        end +
        sum(vertices(lattice)) do i
            return U*interaction{i} - mu*numbers{i}
        end
    end
end

"""
    extended_hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, lattice=InfiniteChain(1); t=1.0, U=1.0, mu=0.0, filling=(1,1))
    fℤ₂ × SU(2) × U(1) single-band Hubbard model
"""
function extended_hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, lattice::MLattice; t=1.0, U=1.0, V=0.5, mu=0.0, filling=(1,1))
    hoppings = hopping(elt, SU2Irrep, U1Irrep; filling=filling)
    interaction = onsiteCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
    interaction2 = neiborCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
    numbers = number(elt, SU2Irrep, U1Irrep; filling=filling)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return -t*hoppings{i,j} + V*interaction2{i,j}
        end +
        sum(vertices(lattice)) do i
            return U*interaction{i} - mu*numbers{i}
        end
    end
end

"""
    hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep},
                    lattice::CustomLattice; kwargs...)
    fℤ₂ × SU(2) × U(1)  Hubbard model
"""
function hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep},
                        lattice::CustomLattice; t=1.0, t2=0.0, th=0.0, th2=0.0, U=6.0, mu=0.0, filling=(1,1))
    hop1 = cdagc(elt, SU2Irrep, U1Irrep; filling=filling)
    hop2 = ccdag(elt, SU2Irrep, U1Irrep; filling=filling)
    onc = onsiteCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
    num = number(elt, SU2Irrep, U1Irrep; filling=filling)
    terms = hubbard_terms(lattice; hop1=hop1, hop2=hop2, onc=onc, num=num, t=t, t2=t2, th=th, th2=th2, U=U, mu=mu)
    I = ProductSector{Tuple{FermionParity, SU2Irrep, U1Irrep}}
    P, Q = filling
    pspace = Vect[I]((0,0,-P) => 1, (0,0,2*Q-P) => 1, (1,1//2,Q-P) => 1)
    return FiniteMPOHamiltonian(fill(pspace, sum(length,lattice.indices)), terms...)
end

"""
    hubbard(elt::Type{<:Number}, ::Type{SU2Irrep}, lattice::CustomLattice; kwargs...)
    fℤ₂ × SU(2) Hubbard model
        default parameters: t=0.0, t2=0.0, th=0.0, th2=0.0, U=0.0, se=0.0, mu=0.0
"""
function hubbard(elt::Type{<:Number}, ::Type{SU2Irrep},
                        lattice::CustomLattice; kwargs...)
    hop1 = cdagc(elt, SU2Irrep)
    hop2 = ccdag(elt, SU2Irrep)
    onc = onsiteCoulomb(elt, SU2Irrep)
    num = number(elt, SU2Irrep)
    sld = singlet_dagger(elt, SU2Irrep; side=:L)
    sl = singlet(elt, SU2Irrep; side=:L)
    terms = hubbard_terms(lattice; hop1=hop1, hop2=hop2, onc=onc, num=num, sld=sld, sl=sl, kwargs...)
    pspace = Vect[(FermionParity ⊠ SU2Irrep)]((0, 0) => 2, (1, 1/2) => 1)
    return FiniteMPOHamiltonian(fill(pspace, sum(length,lattice.indices)), terms...)
end

function hubbard_terms(lattice::CustomLattice; hop1, hop2, onc, num, sld=0, sl=0,
                        t=0.0, t2=0.0, th=0.0, th2=0.0, U=0.0, se=0.0, mu=0.0)
    terms = []
    if length(lattice.lattice[1]) == 3
        tb = twosite_bonds(lattice, 1, 1; intralayer=true, neighbors=Neighbors(1=>Neighbors(lattice.lattice, 2)[1]))
        tb2 = twosite_bonds(lattice, 1, 1; intralayer=true, neighbors=Neighbors(2=>Neighbors(lattice.lattice, 2)[2]))
        for i in eachindex(tb)
            push!(terms, tb[i]=>t*hop1+t'*hop2)
        end
        if !iszero(t2)
            for i in eachindex(tb2)
                push!(terms, tb2[i]=>t2*hop1+t2'*hop2)
            end
        end
        tf = twosite_bonds(lattice, 1, 1; intralayer=false, neighbors=Neighbors(1=>Neighbors(lattice.lattice, 2)[1]))
        tf2 = twosite_bonds(lattice, 1, 1; intralayer=false, neighbors=Neighbors(2=>Neighbors(lattice.lattice, 2)[2]))
        for i in eachindex(tf)
            push!(terms, tf[i]=>th*hop1+th'*hop2)
        end
        if !iszero(th2)
            for i in eachindex(tf2)
                push!(terms, tf2[i]=>th2*hop1+th2'*hop2)
            end
        end
    elseif length(lattice.lattice[1]) == 2
        tb = twosite_bonds(lattice, 1, 1; neighbors=Neighbors(1=>Neighbors(lattice.lattice, 2)[1]))
        tb2 = twosite_bonds(lattice, 1, 1; neighbors=Neighbors(2=>Neighbors(lattice.lattice, 2)[2]))
        for i in eachindex(tb)
            push!(terms, tb[i]=>t*hop1+t'*hop2)
        end
        if !iszero(t2)
            for i in eachindex(tb2)
                push!(terms, tb2[i]=>t2*hop1+t2'*hop2)
            end
        end
        if !iszero(se)
            for i in eachindex(tb)
                push!(terms, tb[i]=>se*sl + se'*sld)
            end
        end
    end
    ob = onesite_bonds(lattice, 1)
    for i in eachindex(ob)
        if !iszero(mu)
            push!(terms, ob[i]=>-mu*num)
        end
        push!(terms, ob[i]=>U*onc)
    end
    return terms
end
"""
    hubbard_bilayer_2band(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep},
                    lattice=BilayerSquare(2, 2; norbit=2); kwargs...)
    fℤ₂ × SU(2) × U(1) two-band bilayer square lattice Hubbard model.
    For La3Ni2O7 thin films, kwargs include the parameters:
        tzz10 = -0.126, tzz20 = -0.016, txx10 = -0.466, txx20 = 0.062,
        tzz1z = -0.439, tzz2z = 0.033, txx1z = 0.005, txx2z = 0.0,
        txz10 =  0.229, txz2z = -0.032, tzz40 = -0.014, txx40 = -0.064,
        txz40 = 0.026, txx80 = -0.015, muz = 0.351, mux = 0.870,
        Uz = 3.51, Ux = 4.03, Up = 2.65, J = 0.56, Vz = 1.31, Vx = 1.04, Vxz = 1.13,
        J1 = -0.56*2, J2 = 0.56, UpJ2 = 2.65 - 0.56/2
    and the bulk La3Ni2O7  parameters are as follow:
        tzz10 = -0.11, tzz20 = -0.017, txx10 = -0.483, txx20 = 0.069,
        tzz1z = -0.635, tzz2z = 0.0, txx1z = 0.0, txx2z = 0.0,
        txz10 = 0.239, txz2z = -0.034, tzz40 = 0.0, txx40 = 0.0,
        txz40 = 0.0, txx80 = 0.0, muz = 0.409, mux = 0.776,
        Uz = 3.7, Ux = 3.7, Up = 2.5, J = 0.6, Vz = 0.0, Vx = 0.0, Vxz = 0.0,
        J1 = -0.6*2, J2 = 0.6, UpJ2 = 2.5 - 0.6/2
"""
function hubbard_bilayer_2band(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep},
                        lattice=BilayerSquare(2, 2; norbit=2); filling=(3,4), kwargs...)
    hop = hopping(elt, SU2Irrep, U1Irrep; filling=filling)
    onc = onsiteCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
    num = number(elt, SU2Irrep, U1Irrep; filling=filling)
    nbc = neiborCoulomb(elt, SU2Irrep, U1Irrep; filling=filling)
    sf = heisenberg(elt, SU2Irrep, U1Irrep; filling=filling)
    ph = pairhopping(elt, SU2Irrep, U1Irrep; filling=filling)
    terms = hubbard_bilayer_2band_terms(lattice; hop=hop, onc=onc, num=num, nbc=nbc, sf=sf, ph=ph, kwargs...)
    I = ProductSector{Tuple{FermionParity, SU2Irrep, U1Irrep}}
    P, Q = filling
    pspace = Vect[I]((0,0,-P) => 1, (0,0,2*Q-P) => 1, (1,1//2,Q-P) => 1)
    return FiniteMPOHamiltonian(fill(pspace, sum(length,lattice.indices)), terms...)
end

function hubbard_bilayer_2band(elt::Type{<:Number}, ::Type{SU2Irrep},
                        lattice=BilayerSquare(2, 2; norbit=2); kwargs...)
    hop = hopping(elt, SU2Irrep)
    onc = onsiteCoulomb(elt, SU2Irrep)
    num = number(elt, SU2Irrep)
    nbc = neiborCoulomb(elt, SU2Irrep)
    sf = heisenberg(elt, SU2Irrep)
    ph = pairhopping(elt, SU2Irrep)
    sld = singlet_dagger(elt, SU2Irrep; side=:L)
    sl = singlet(elt, SU2Irrep; side=:L)
    terms = hubbard_bilayer_2band_terms(lattice; hop=hop, onc=onc, num=num, nbc=nbc, sf=sf, ph=ph, sld=sld, sl=sl, kwargs...)
    pspace = Vect[(FermionParity ⊠ SU2Irrep)]((0, 0) => 2, (1, 1/2) => 1)
    return FiniteMPOHamiltonian(fill(pspace, sum(length,lattice.indices)), terms...)
end

function hubbard_bilayer_2band_terms(lattice; hop, onc, num, nbc, sf, ph, sld=0, sl=0,
                        tzz10=0.0, tzz20=0.0, txx10=0.0, txx20=0.0, tzz1z=0.0, tzz2z=0.0, txx1z=0.0, txx2z=0.0, txz10=0.0, txz2z=0.0, tzz40=0.0, txx40=0.0, txz40=0.0, txx80=0.0,
                        muz=0.0, mux=0.0, Uz=0.0, Ux=0.0, Up=0.0, J=0.0, Vz=0.0, Vx=0.0, Vxz=0.0, J1=0.0, J2=0.0, UpJ2=0.0, spmz=0.0, spmx=0.0)
    terms = []
    if !iszero(tzz10)
        zz10 = twosite_bonds(lattice, 1, 1; intralayer=true, neighbors=Neighbors(1=>1))
        for i in eachindex(zz10)
            push!(terms, zz10[i]=>tzz10*hop)
        end
    end
    if !iszero(tzz20)
        zz20 = twosite_bonds(lattice, 1, 1; intralayer=true, neighbors=Neighbors(2=>sqrt(2)))
        for i in eachindex(zz20)
            push!(terms, zz20[i]=>tzz20*hop)
        end

    end
    if !iszero(txx10)
        xx10 = twosite_bonds(lattice, 2, 2; intralayer=true, neighbors=Neighbors(1=>1))
        for i in eachindex(xx10)
            push!(terms, xx10[i]=>txx10*hop)
        end
    end
    if !iszero(txx20)
        xx20 = twosite_bonds(lattice, 2, 2; intralayer=true, neighbors=Neighbors(2=>sqrt(2)))
        for i in eachindex(xx20)
            push!(terms, xx20[i]=>txx20*hop)
        end
    end
    if !iszero(tzz1z)
        zz1z = twosite_bonds(lattice, 1, 1; intralayer=false, neighbors=Neighbors(1=>1))
        for i in eachindex(zz1z)
            push!(terms, zz1z[i]=>tzz1z*hop)
        end
    end
    if !iszero(tzz2z)
        zz2z = twosite_bonds(lattice, 1, 1; intralayer=false, neighbors=Neighbors(2=>sqrt(2)))
        for i in eachindex(zz2z)
            push!(terms, zz2z[i]=>tzz2z*hop)
        end
    end
    if !iszero(txx1z)
        xx1z = twosite_bonds(lattice, 2, 2; intralayer=false, neighbors=Neighbors(1=>1))
        for i in eachindex(xx1z)
            push!(terms, xx1z[i]=>txx1z*hop)
        end
    end
    if !iszero(txx2z)
        xx2z = twosite_bonds(lattice, 2, 2; intralayer=false, neighbors=Neighbors(2=>sqrt(2)))
        for i in eachindex(xx2z)
            push!(terms, xx2z[i]=>txx2z*hop)
        end
    end
    if !iszero(txz10)
        xz10 = [twosite_bonds(lattice, 1, 2; intralayer=true, neighbors=Neighbors(1=>1)); twosite_bonds(lattice, 2, 1; intralayer=true, neighbors=Neighbors(1=>1))]
        for i in eachindex(xz10)
            a, b = find_position(lattice.indices, xz10[i][1]), find_position(lattice.indices, xz10[i][2])
            if (lattice.lattice[a] - lattice.lattice[b])[2] ≈ 0
                push!(terms, xz10[i]=>txz10*hop)
            elseif (lattice.lattice[a] - lattice.lattice[b])[1] ≈ 0
                push!(terms, xz10[i]=>-txz10*hop)
            else
                throw(ArgumentError("Invalid n1 xz bond"))
            end
        end
    end
    if !iszero(txz2z)
        xz2z = [twosite_bonds(lattice, 1, 2; intralayer=false, neighbors=Neighbors(2=>sqrt(2))); twosite_bonds(lattice, 2, 1; intralayer=false, neighbors=Neighbors(2=>sqrt(2)))]
        for i in eachindex(xz2z)
            a, b = find_position(lattice.indices, xz2z[i][1]), find_position(lattice.indices, xz2z[i][2])
            if (lattice.lattice[a] - lattice.lattice[b])[2] ≈ 0
                push!(terms, xz2z[i]=>txz2z*hop)
            elseif (lattice.lattice[a] - lattice.lattice[b])[1] ≈ 0
                push!(terms, xz2z[i]=>-txz2z*hop)
            else
                throw(ArgumentError("Invalid n2 xz bond"))
            end
        end
    end
    if !iszero(tzz40)
        zz40 = twosite_bonds(lattice, 1, 1; intralayer=true, neighbors=Neighbors(4=>2.0))
        for i in eachindex(zz40)
            push!(terms, zz40[i]=>tzz40*hop)
        end
    end
    if !iszero(txx40)
        xx40 = twosite_bonds(lattice, 2, 2; intralayer=true, neighbors=Neighbors(4=>2.0))
        for i in eachindex(xx40)
            push!(terms, xx40[i]=>txx40*hop)
        end
    end
    if !iszero(txz40)
        xz40 = [twosite_bonds(lattice, 1, 2; intralayer=true, neighbors=Neighbors(4=>2.0)); twosite_bonds(lattice, 2, 1; intralayer=true, neighbors=Neighbors(4=>2.0))]
        for i in eachindex(xz40)
            a, b = find_position(lattice.indices, xz40[i][1]), find_position(lattice.indices, xz40[i][2])
            if (lattice.lattice[a] - lattice.lattice[b])[2] ≈ 0
                push!(terms, xz40[i]=>txz40*hop)
            elseif (lattice.lattice[a] - lattice.lattice[b])[1] ≈ 0
                push!(terms, xz40[i]=>-txz40*hop)
            else
                throw(ArgumentError("Invalid n4 xz bond"))
            end
        end
    end
    if !iszero(txx80)
        xx80 = twosite_bonds(lattice, 2, 2; intralayer=true, neighbors=Neighbors(8=>3.0))
        for i in eachindex(xx80)
            push!(terms, xx80[i]=>txx80*hop)
        end
    end
    a = onesite_bonds(lattice, 1)
    for i in eachindex(a)
        push!(terms, a[i]=>muz*num)
        if !iszero(Uz)
            push!(terms, a[i]=>Uz*onc)
        end
    end
    b = onesite_bonds(lattice, 2)
    for i in eachindex(b)
        push!(terms, b[i]=>mux*num)
        if !iszero(Ux)
            push!(terms, b[i]=>Ux*onc)
        end
    end
    ab = onesite_bonds(lattice, 1, 2)
    for i in eachindex(ab)
        if !iszero(UpJ2)
            push!(terms, ab[i]=>UpJ2*nbc)
        end
        if !iszero(J1)
            push!(terms, ab[i]=>J1*sf)
        end
        if !iszero(J2)
            push!(terms, ab[i]=>J2*ph)
        end
    end
    if !iszero(Vz)
        zz1z = twosite_bonds(lattice, 1, 1; intralayer=false, neighbors=Neighbors(1=>1))
        for i in eachindex(zz1z)
            push!(terms, zz1z[i]=>Vz*nbc)
        end
    end
    if !iszero(Vx)
        xx1z = twosite_bonds(lattice, 2, 2; intralayer=false, neighbors=Neighbors(1=>1))
        for i in eachindex(xx1z)
            push!(terms, xx1z[i]=>Vx*nbc)
        end
    end
    if !iszero(Vxz)
        zx1z = [twosite_bonds(lattice, 1, 2; intralayer=false, neighbors=Neighbors(1=>1)); twosite_bonds(lattice, 2, 1; intralayer=false, neighbors=Neighbors(1=>1))]
        for i in eachindex(zx1z)
            push!(terms, zx1z[i]=>Vxz*nbc)
        end
    end
    if !iszero(spmz)
        zz1z = twosite_bonds(lattice, 1, 1; intralayer=false, neighbors=Neighbors(1=>1))
        for i in eachindex(zz1z)
            push!(terms, zz1z[i]=>spmz*sl + spmz'*sld)
        end
    end
    if !iszero(spmx)
        xx1z = twosite_bonds(lattice, 2, 2; intralayer=false, neighbors=Neighbors(1=>1))
        for i in eachindex(xx1z)
            push!(terms, xx1z[i]=>spmx*sl + spmx'*sld)
        end
    end
    return terms
end
"""
    kitaev_hubbard(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, lattice=FiniteChain(1); t=1.0, tz=0.0, U=1.0, mu=0.0, filling=(1,1))

Construct a fℤ₂ × U(1) × U(1) 1d-Chain Kitaev-Hubbard model without pairing terms.
"""
function kitaev_hubbard(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, lattice=FiniteChain(1); t=1.0, tz=0.0, U=1.0, mu=0.0, filling=(1,1))
    hoppings = hopping(elt, U1Irrep, U1Irrep;filling=filling)
    σz_hoppings = σz_hopping(elt, U1Irrep, U1Irrep;filling=filling)
    interaction = onsiteCoulomb(elt, U1Irrep, U1Irrep; filling=filling)
    numbers = number(elt, U1Irrep, U1Irrep; filling=filling)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return (-t*hoppings{i,j} - tz*σz_hoppings{i,j})/2
        end +
        sum(vertices(lattice)) do i
            return U*interaction{i} - mu*numbers{i}
        end
    end
end

"""
    heisenberg_model(elt::Type{<:Number}, ::Type{SU2Irrep}, lattice=FiniteChain(1); J=1.0)
"""
function heisenberg_model(elt::Type{<:Number}, ::Type{SU2Irrep}, lattice=FiniteChain(1); J=1.0,  spin=1//2)
    hei = heisenberg(elt, SU2Irrep, spin)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return J * hei{i, j}
        end
    end
end

"""
    JKGGp_model(L, x_indices, y_indices, z_indices; spin=1//2, J=1, K=0, G=0, Gp=0)

Construct a JKΓΓ' (Kitaev-Heisenberg-Gamma) spin model Hamiltonian without symmetry constraints.

The Hamiltonian on each bond type (x, y, z) is:
``H_\\gamma = (J+K) S^\\gamma_i S^\\gamma_j + J \\sum_{\\alpha \\neq \\gamma} S^\\alpha_i S^\\alpha_j + \\Gamma(\\ldots) + \\Gamma'(\\ldots)``

# Arguments
- `L`: total number of sites.
- `x_indices`, `y_indices`, `z_indices`: arrays of `(site_i, site_j)` tuples for x-, y-, z-type bonds.
- `spin`: spin quantum number (default: 1/2).
- `J`: Heisenberg coupling.
- `K`: Kitaev coupling.
- `G` (`Γ`): off-diagonal symmetric exchange.
- `Gp` (`Γ'`): off-diagonal symmetric exchange (primed).
"""
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
