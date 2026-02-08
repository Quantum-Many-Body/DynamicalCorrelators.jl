"""
    abstract type CustomLattice end

Abstract supertype for all custom lattice structures used in DynamicalCorrelators.jl.

Subtypes must provide fields `lattice` (a `QLattice` from QuantumLattices) and `indices`
(a nested array mapping each lattice site to its orbital-resolved MPS site indices).
"""
abstract type CustomLattice end

"""
    BilayerSquare{T,U,L} <: CustomLattice

A bilayer square lattice with two layers stacked along the z-direction.

# Fields
- `W::T`: width of the lattice (number of sites along x).
- `L::T`: length of the lattice (number of sites along y).
- `unitcell::U`: the two-site unit cell (one site per layer).
- `lattice::L`: the full lattice generated from the unit cell.
- `indices::AbstractArray`: mapping from lattice site index to MPS site indices, grouped by orbital.
"""
struct BilayerSquare{T<:Integer, U<:QLattice, L<:QLattice} <: CustomLattice
    W::T
    L::T
    unitcell::U
    lattice::L
    indices::AbstractArray{<:AbstractArray{T, 1}, 1}
end

"""
    BilayerSquare(W::Integer, L::Integer; norbit=2, periodic=false)

Construct a `BilayerSquare` lattice of size `W × L` with `norbit` orbitals per site.

# Arguments
- `W`: width (number of unit cells along x).
- `L`: length (number of unit cells along y).
- `norbit`: number of orbitals per site (default: 2).
- `periodic`: if `true`, periodic boundary conditions along x; otherwise open (default: `false`).
"""
function BilayerSquare(W::Integer, L::Integer;
                    norbit::Integer=2,
                    periodic::Bool=false)
    unitcell = Lattice([0,0,0],[0,0,1]; vectors=[[1,0,0],[0,1,0]], name=:BS)
    lattice = periodic ? Lattice(unitcell, (W, L), ('p', 'o')) : Lattice(unitcell, (W, L), ('o', 'o'))
    indeces = [[(i-1)*norbit + j for j in 1:norbit] for i in 1:length(lattice)]
    return BilayerSquare(W, L, unitcell, lattice, indeces)
end

"""
    twosite_bonds(customlattice::CustomLattice, a::Integer, b::Integer; intralayer=true, neighbors=Neighbors(1=>1))

Return an array of `(site_a, site_b)` pairs representing two-site bonds on the lattice.

For 3D lattices (bilayer), `intralayer` controls whether to select bonds within the same
layer (`true`) or between layers (`false`). Orbital indices `a` and `b` select which orbital
on each site to include.

# Arguments
- `customlattice`: the lattice structure.
- `a`, `b`: orbital indices for the two sites of each bond.
- `intralayer`: select intralayer (`true`) or interlayer (`false`) bonds (only relevant for 3D lattices).
- `neighbors`: neighbor specification from QuantumLattices.
"""
function twosite_bonds(customlattice::CustomLattice, a::Integer, b::Integer;
    intralayer::Bool=true,
    neighbors::Neighbors=Neighbors(1=>1))
    if length(customlattice.lattice[1]) == 2
        bs = bonds(customlattice.lattice, neighbors)
    elseif length(customlattice.lattice[1]) == 3
        bs = intralayer ? filter(bond->bond.points[1].rcoordinate[3] == bond.points[2].rcoordinate[3], bonds(customlattice.lattice, neighbors)) : filter(bond->bond.points[1].rcoordinate[3] !== bond.points[2].rcoordinate[3], bonds(customlattice.lattice, neighbors))
    end
    tbs = Vector(undef, length(bs))
    for (o, bond) in enumerate(bs)
    (i, j) = bond.points[1].site > bond.points[2].site ? (bond.points[2].site, bond.points[1].site) : (bond.points[1].site, bond.points[2].site)   
    tbs[o] = (customlattice.indices[i][a], customlattice.indices[j][b])
    end
    return tbs
end

"""
    onesite_bonds(customlattice::CustomLattice, a::Integer, b::Integer)

Return an array of `(site_a, site_b)` tuples for on-site (two-orbital) terms,
where `a` and `b` are the orbital indices on the same lattice site.
"""
function onesite_bonds(customlattice::CustomLattice, a::Integer, b::Integer)
    bs =  bonds(customlattice.lattice, 0)
    obs = Vector(undef, length(bs))
    for i in eachindex(bs)
    obs[i] = (customlattice.indices[i][a], customlattice.indices[i][b])
    end
    return obs
end

"""
    onesite_bonds(customlattice::CustomLattice, a::Integer)

Return an array of single-element tuples `(site_a,)` for on-site (single-orbital) terms.
"""
function onesite_bonds(customlattice::CustomLattice, a::Integer)
    bs =  bonds(customlattice.lattice, 0)
    obs = Vector(undef, length(bs))
    for i in eachindex(bs)
    obs[i] = (customlattice.indices[i][a],)
    end
    return obs
end

"""
    find_position(indices, target::Integer)

Find the lattice site index that contains the MPS site index `target`.
Returns `nothing` if not found.
"""
function find_position(indices::AbstractArray{<:AbstractArray{<:Integer, 1}, 1}, target::Integer)
    for (i, inner) in enumerate(indices)
        if target in inner
            return i
        end
    end
    return nothing 
end


"""
    Square{T,U,L} <: CustomLattice

A 2D square lattice.

# Fields
- `W::T`: width of the lattice.
- `L::T`: length of the lattice.
- `unitcell::U`: the single-site unit cell.
- `lattice::L`: the full lattice.
- `indices`: mapping from lattice site to MPS site indices.
"""
struct Square{T<:Integer, U<:QLattice, L<:QLattice} <: CustomLattice
    W::T
    L::T
    unitcell::U
    lattice::L
    indices::AbstractArray{<:AbstractArray{T, 1}, 1}
end

"""
    Square(W::Integer, L::Integer; norbit=1, periodic=false)

Construct a `Square` lattice of size `W × L`.

# Arguments
- `norbit`: number of orbitals per site (default: 1).
- `periodic`: if `true`, periodic boundary along x (default: `false`).
"""
function Square(W::Integer, L::Integer;
                    norbit::Integer=1,
                    periodic::Bool=false)
    unitcell = Lattice([0,0]; vectors=[[1,0],[0,1]], name=:S)
    lattice = periodic ? Lattice(unitcell, (W, L), ('p', 'o')) : Lattice(unitcell, (W, L), ('o', 'o'))
    indeces = [[(i-1)*norbit + j for j in 1:norbit] for i in 1:length(lattice)]
    return Square(W, L, unitcell, lattice, indeces)
end


"""
    Custom{T,L} <: CustomLattice

A lattice wrapper for arbitrary `QLattice` objects from QuantumLattices.

# Fields
- `lattice::L`: the underlying QuantumLattices lattice.
- `indices`: mapping from lattice site to MPS site indices.
"""
struct Custom{T<:Integer, L<:QLattice} <: CustomLattice
    lattice::L
    indices::AbstractArray{<:AbstractArray{T, 1}, 1}
end

"""
    Custom(lattice::QLattice; norbit=1)

Wrap a `QLattice` as a `Custom` lattice with `norbit` orbitals per site.
"""
function Custom(lattice::QLattice; norbit=1)
    indeces = [[(i-1)*norbit + j for j in 1:norbit] for i in 1:length(lattice)]
    return Custom(lattice, indeces)
end


"""
    snake_2D(directions, orders)

Generate a 2D lattice path (snake-like) from a sequence of directional steps.

# Arguments
- `directions`: array of direction vectors (e.g., `[[1,0], [0,1]]`).
- `orders`: array of integers specifying which direction to take at each step.
  Positive values move forward, negative values move backward along that direction.

# Returns
A vector of 2D coordinates tracing the path starting from `[0, 0]`.
"""
function snake_2D(directions, orders)
    lattice = Vector{Vector}(undef, length(orders)+1)
    lattice[1] = [0, 0]
    for i in eachindex(orders) 
        if orders[i] > 0
            lattice[i+1] = lattice[i] + directions[orders[i]]
        else
            lattice[i+1] = lattice[i] - directions[abs(orders[i])]
        end
    end
    return lattice
end

"""
    kitaev_bonds(directions, lattice)

Classify nearest-neighbor bonds of a lattice into three groups based on their
alignment with three given directions (for Kitaev-type models).

Each bond is assigned to the direction whose unit vector is parallel to the bond vector
(using the dot product criterion `|a·b| ≈ |a||b|`).

# Arguments
- `directions`: array of 3 direction vectors defining x-, y-, z-type bonds.
- `lattice`: a `QLattice` from QuantumLattices.

# Returns
Three arrays of `(site_i, site_j)` tuples, one for each direction.
"""
function kitaev_bonds(directions, lattice)
    bs = bonds(lattice, Neighbors(1=>1))
    indices₁, indices₂, indices₃ = [], [], []
    for bond in bs
        bd = bond.points[1].rcoordinate - bond.points[2].rcoordinate
        if abs(dot(bd,directions[1])) ≈ norm(bd)*norm(directions[1])
            (a, b) = bond.points[1].site > bond.points[2].site ? (bond.points[2].site, bond.points[1].site) : (bond.points[1].site, bond.points[2].site)
            push!(indices₁, (a, b))
        elseif abs(dot(bd,directions[2])) ≈ norm(bd)*norm(directions[2])
            (a, b) = bond.points[1].site > bond.points[2].site ? (bond.points[2].site, bond.points[1].site) : (bond.points[1].site, bond.points[2].site)
            push!(indices₂, (a, b))
        elseif abs(dot(bd,directions[3])) ≈ norm(bd)*norm(directions[3])
            (a, b) = bond.points[1].site > bond.points[2].site ? (bond.points[2].site, bond.points[1].site) : (bond.points[1].site, bond.points[2].site)
            push!(indices₃, (a, b))
        else
            throw(ArgumentError("bond is dismatched with any directions"))
        end
    end
    return indices₁, indices₂, indices₃
end