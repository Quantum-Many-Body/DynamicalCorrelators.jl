"""
    hamiltonian(terms, lattice, hilbert; neighbors=nothing, filling=(1,1))

Construct a `FiniteMPOHamiltonian` from QuantumLattices `Term`s on a given `lattice`
with Hilbert space `hilbert`. Uses fℤ₂ × U(1) × U(1) symmetry.

# Arguments
- `terms`: tuple of `Term` objects (e.g., `Hopping`, `Hubbard`, `Onsite`).
- `lattice`: a `Lattice` from QuantumLattices.
- `hilbert`: the Hilbert space specification.
- `neighbors`: neighbor order. If `nothing`, inferred from `terms`.
- `filling`: tuple `(P, Q)` defining the filling fraction.
"""
function hamiltonian(terms::Tuple{Vararg{Term}}, lattice::Lattice, hilbert::Hilbert; neighbors::Union{Nothing, Int, Neighbors}=nothing, filling::NTuple{2, Integer}=(1,1))
    isnothing(neighbors) && (neighbors = maximum(term->term.bondkind, terms))
    bond = bonds(lattice, neighbors)
    operators = expand(OperatorGenerator(bond, hilbert, terms))
    return hamiltonian(operators, length(lattice), filling)
end

"""
    hamiltonian(operators::OperatorSet, len, filling)

Convert a set of symbolic operators into a `FiniteMPOHamiltonian` with U(1)×U(1) symmetry.
Each operator is converted to an MPO tensor via `_convert_operator`.
"""
function hamiltonian(operators::OperatorSet{<:Operator}, len::Integer, filling::NTuple{2, Integer})
    mpos = Vector(undef, length(operators))
    for (i, op) in enumerate(operators)
        mpos[i] = _convert_operator(op, filling)
    end
    I = ProductSector{Tuple{FermionParity, U1Irrep, U1Irrep}}
    P, Q = filling
    pspace = Vect[I]((0,0,-P)=>1, (0,0,2*Q-P)=>1, (1,1,Q-P)=>1, (1,-1,Q-P)=>1)
    return FiniteMPOHamiltonian(fill(pspace, len), mpos...)
end

# Convert a 2-index symbolic operator to an MPO site term.
# Handles both on-site (single-site) and two-site cases, determining the
# tensor representation from the spin/nambu indices.
function _convert_operator(op::Operator{<:Number, <:NTuple{2, CoordinatedIndex}}, filling::NTuple{2, Integer})
    value = op.value
    sites = unique([op.id[i].index.site for i in 1:2])
    if length(sites) == 1
        mpoop = contract_onesite(_index2tensor(typeof(value), op.id[1].index, :L, filling), _index2tensor(typeof(value), op.id[2].index, :R, filling))
        return (sites[1], ) => value*mpoop
    elseif length(sites) == 2
        if sites[1] < sites[2]
            mpoop = contract_twosite(_index2tensor(typeof(value), op.id[1].index, :L, filling), _index2tensor(typeof(value), op.id[2].index, :R, filling))
            return (sites[1], sites[2]) => value*mpoop
        else
            mpoop = contract_twosite(_index2tensor(typeof(value), op.id[2].index, :L, filling), _index2tensor(typeof(value), op.id[1].index, :R, filling))
            return (sites[2], sites[1]) => value*mpoop
        end
    end
end

# Convert a 4-index symbolic operator to an MPO site term.
# Handles on-site (1 site) and two-site (2 sites) cases for 4-body terms
# like pair hopping or Hund's coupling.
function _convert_operator(op::Operator{<:Number, <:NTuple{4, CoordinatedIndex}}, filling::NTuple{2, Integer})
    value = op.value
    sites = unique([op.id[i].index.site for i in 1:4])
    if length(sites) == 1
        mpoop = contract_onesite(contract_onesite(_index2tensor(typeof(value), op.id[1].index, :L, filling), _index2tensor(typeof(value), op.id[2].index, :R, filling)), contract_onesite(_index2tensor(typeof(value), op.id[3].index, :L, filling), _index2tensor(typeof(value), op.id[4].index, :R, filling)))
        return (sites[1], ) => value*mpoop
    elseif length(sites) == 2
        if sites[1] < sites[2]
            mpoop = contract_twosite(contract_onesite(_index2tensor(typeof(value), op.id[1].index, :L, filling), _index2tensor(typeof(value), op.id[2].index, :R, filling)), contract_onesite(_index2tensor(typeof(value), op.id[3].index, :L, filling), _index2tensor(typeof(value), op.id[4].index, :R, filling)))
            return (sites[1], sites[2]) => value*mpoop
        else
            mpoop = contract_twosite(contract_onesite(_index2tensor(typeof(value), op.id[4].index, :L, filling), _index2tensor(typeof(value), op.id[3].index, :R, filling)), contract_onesite(_index2tensor(typeof(value), op.id[2].index, :L, filling), _index2tensor(typeof(value), op.id[1].index, :R, filling)))
            return (sites[2], sites[1]) => value*mpoop
        end
    end
end

# Map a QuantumLattices Index to the corresponding fermionic tensor operator.
# Determines creation (nambu=2) vs annihilation, and spin-up (1/2) vs spin-down.
function _index2tensor(elt::Type{<:Number}, ids::Index, side::Symbol, filling::NTuple{2, Integer})
    ids.internal.nambu == 2 ? ten = e_plus : ten = e_min
    ids.internal.spin == 1//2 ? spin = :up : spin = :down
    return ten(elt, U1Irrep, U1Irrep; side=side, spin=spin, filling=filling)
end
