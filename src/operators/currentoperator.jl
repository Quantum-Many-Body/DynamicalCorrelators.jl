"""
    j_l(ops, site::Integer, direction::Vector)
    Operators like local current operator are depended on the specific forms of Hamiltonian and lattice, for example, j_l = ∂N_l(t)/∂t = i/ħ * [H, N_l(t)] = i * ∑_k t_lk * c^†_l c_k * (r_l - r_k)
    WARNING: In the following function, we remove the "i" for convenience, and add "-1" in the final result of a current-current propagator as compensation. 
"""
function j_l(ops, len::Integer, site::Integer, direction::Vector; filling::NTuple{2, Integer}=(1,1))
    lops = filter(op -> (length(op)==2)&&((op.id[1].index.site)!==(op.id[2].index.site))&&((op.id[1].index.site==site)||(op.id[2].index.site==site)), collect(ops))
    mpos = Vector(undef, length(lops))
    for (i, op) in enumerate(lops)
        value = op.value
        sites = [op.id[i].index.site for i in 1:2]
        distance = op.id[1].rcoordinate - op.id[2].rcoordinate
        if sites[1] < sites[2]
            mpoop = contract_twosite(_index2tensor(typeof(value), op.id[1].index, :L, filling), _index2tensor(typeof(value), op.id[2].index, :R, filling))
            mpos[i] = (sites[1], sites[2]) => value*(dot(distance, direction)/norm(direction))*mpoop
        else
            mpoop = contract_twosite(_index2tensor(typeof(value), op.id[2].index, :L, filling), _index2tensor(typeof(value), op.id[1].index, :R, filling))
            mpos[i] = (sites[2], sites[1]) => value*(dot(distance, direction)/norm(direction))*mpoop
        end
    end
    I = ProductSector{Tuple{FermionParity, U1Irrep, U1Irrep}}
    P, Q = filling
    pspace = Vect[I]((0,0,-P)=>1, (0,0,2*Q-P)=>1, (1,1,Q-P)=>1, (1,-1,Q-P)=>1)
    return MPOHamiltonian(fill(pspace, len), mpos...)
end

