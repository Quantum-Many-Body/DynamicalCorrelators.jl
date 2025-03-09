#pre-defined lattices

struct BilayerSquare
    W::Integer
    L::Integer
    norbit::Integer
    lattice::Lattice
end

function BilayerSquare(W::Integer, L::Integer; norbit::Integer=2, periodic::Bool=true)
    unitcell = Lattice([0, 0, 0], [0, 0, 1]; vectors = [[1, 0, 0], [0, 1, 0]], name=:BS)
    lattice = periodic ? Lattice(unitcell, (W, L), ('p', 'o')) : Lattice(unitcell, (W, L), ('o', 'o'))
    return BilayerSquare(W, L, norbit, lattice)
end