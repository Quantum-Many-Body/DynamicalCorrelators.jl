#pre-defined lattices

abstract type CustomLattice end

struct BilayerSquare{T<:Integer, U<:QLattice, L<:QLattice} <: CustomLattice
    W::T
    L::T
    unitcell::U
    lattice::L
    indices::AbstractArray{<:AbstractArray{T, 1}, 1}
end

function BilayerSquare(W::Integer, L::Integer; 
                    norbit::Integer=2, 
                    periodic::Bool=false)
    unitcell = Lattice([0,0,0],[0,0,1]; vectors=[[1,0,0],[0,1,0]], name=:BS)
    lattice = periodic ? Lattice(unitcell, (W, L), ('p', 'o')) : Lattice(unitcell, (W, L), ('o', 'o'))
    indeces = [[(i-1)*norbit + j for j in 1:norbit] for i in 1:length(lattice)]
    return BilayerSquare(W, L, unitcell, lattice, indeces)
end

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

function onesite_bonds(customlattice::CustomLattice, a::Integer, b::Integer)
    bs =  bonds(customlattice.lattice, 0)
    obs = Vector(undef, length(bs))
    for i in eachindex(bs) 
    obs[i] = (customlattice.indices[i][a], customlattice.indices[i][b])
    end
    return obs
end

function onesite_bonds(customlattice::CustomLattice, a::Integer)
    bs =  bonds(customlattice.lattice, 0)
    obs = Vector(undef, length(bs))
    for i in eachindex(bs) 
    obs[i] = (customlattice.indices[i][a],)
    end
    return obs
end

function find_position(indices::AbstractArray{<:AbstractArray{<:Integer, 1}, 1}, target::Integer)
    for (i, inner) in enumerate(indices)
        if target in inner
            return i
        end
    end
    return nothing 
end


struct Square{T<:Integer, U<:QLattice, L<:QLattice} <: CustomLattice
    W::T
    L::T
    unitcell::U
    lattice::L
    indices::AbstractArray{<:AbstractArray{T, 1}, 1}
end

function Square(W::Integer, L::Integer; 
                    norbit::Integer=1, 
                    periodic::Bool=false)
    unitcell = Lattice([0,0]; vectors=[[1,0],[0,1]], name=:S)
    lattice = periodic ? Lattice(unitcell, (W, L), ('p', 'o')) : Lattice(unitcell, (W, L), ('o', 'o'))
    indeces = [[(i-1)*norbit + j for j in 1:norbit] for i in 1:length(lattice)]
    return Square(W, L, unitcell, lattice, indeces)
end


struct Custom{T<:Integer, L<:QLattice} <: CustomLattice
    lattice::L
    indices::AbstractArray{<:AbstractArray{T, 1}, 1}
end

function Custom(lattice::QLattice; norbit=1)
    indeces = [[(i-1)*norbit + j for j in 1:norbit] for i in 1:length(lattice)]
    return Custom(lattice, indeces)
end


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