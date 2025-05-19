"""

"""
abstract type AbstractCorrelation end

struct PairCorrelation{K} <: AbstractCorrelation
    operator::AbstractTensorMap
    lattice:: CustomLattice
    amplitudes::AbstractArray
    indices::AbstractArray
end

function PairCorrelation{K}(operator::AbstractTensorMap, latt::CustomLattice, neighbors::Neighbors, a::Integer, b::Integer; amplitude::Union{Nothing, Function}=nothing, intralayer::Union{Nothing, Bool}=nothing) where K
    operator = operator
    lattice = latt
    amplitudes, indices = pair_amplitude_indices(latt, neighbors, a, b; amplitude=amplitude, intralayer=intralayer)
    return PairCorrelation{K}(operator, lattice, amplitudes, indices)
end

function pair_amplitude_indices(latt::CustomLattice, neighbors::Neighbors, a::Integer, b::Integer; amplitude::Union{Nothing, Function}=nothing, intralayer::Union{Nothing, Bool}=nothing)
    bs = isnothing(intralayer) ? bonds(latt.lattice, neighbors) : intralayer ? filter(bond->bond.points[1].rcoordinate[3] == bond.points[2].rcoordinate[3], bonds(latt.lattice, neighbors)) : filter(bond->bond.points[1].rcoordinate[3] !== bond.points[2].rcoordinate[3], bonds(latt.lattice, neighbors))
    amp = Vector{Vector}(undef, length(latt.lattice))
    indices = Vector{Vector{Vector{Int}}}(undef, length(latt.lattice))
    for i in 1:length(latt.lattice)
        ibs = filter(bo -> any(p -> p.site == i, bo.points), bs)
        pos = [collect(p.site for p in bo.points) for bo in ibs]
        amp[i] = isnothing(amplitude) ? [1.0 for _ in 1:length(ibs)] : [amplitude(b) for b in ibs]
        if a == b
            indices[i] = map(p -> map(s -> latt.indices[s][a], p), pos)
        else
            indices[i] = [latt.indices[p][a], latt.indices[p][b] for p in pos]
        end
    end
    return amp, indices
end

function correlator(correlation::AbstractCorrelation, gs::AbstractFiniteMPS; is=Vector((length(correlation.lattice.lattice)÷2):-1:1), js=Vector((length(correlation.lattice.lattice)÷2+1):1:length(correlation.lattice.lattice)))
    correlator(correlation.operator, gs, correlation.amplitudes, correlation.indices, is, js)
end

function correlator(O::AbstractTensorMap, gs::AbstractFiniteMPS, amplitudes::AbstractArray, indices::AbstractArray, is::AbstractArray{<:Integer, 1}, js::AbstractArray{<:Integer, 1})
    @assert length(amplitudes) == length(indices) "Length of amplitudes and indices must be the same"
    @assert length(is) == length(js) "Length of is and js must be the same"
    Fr = zeros(Float64, length(is))
    for i in eachindex(is) 
            Fr[i] = sum(dot(chargedMPS(O, gs, Tuple(indices[is[i]][a])), chargedMPS(O, gs, Tuple(indices[js[i]][b]))) * dot(amplitudes[is[i]][a], amplitudes[js[i]][b]) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
    end
    return Fr
end

function correlator(O::AbstractTensorMap, gs::AbstractFiniteMPS, i::NTuple{1, Integer}, j::NTuple{1, Integer})
    dot(chargedMPS(O, gs, i[1]), chargedMPS(O, gs, j[1]))
end

function correlator(O::AbstractTensorMap, gs::AbstractFiniteMPS, i::NTuple{2, Integer}, j::NTuple{2, Integer})
    dot(chargedMPS(O, gs, i[1], i[2]), chargedMPS(O, gs, j[1], j[2]))
end