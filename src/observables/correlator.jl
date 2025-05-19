"""
    abstract type AbstractCorrelation end
"""
abstract type AbstractCorrelation end

"""
    PairCorrelation{K} <: AbstractCorrelation
"""
struct PairCorrelation{K} <: AbstractCorrelation
    operator::AbstractTensorMap
    lattice::CustomLattice
    amplitudes::AbstractArray
    indices::AbstractArray
end

"""
    PairCorrelation{K}(operator::AbstractTensorMap, latt::CustomLattice, neighbors::Neighbors, a::Integer, b::Integer; amplitude::Union{Nothing, Function}=nothing, intralayer::Union{Nothing, Bool}=nothing) 
"""
function PairCorrelation{K}(operator::AbstractTensorMap, latt::CustomLattice, neighbors::Neighbors, a::Integer, b::Integer; amplitude::Union{Nothing, Function}=nothing, intralayer::Union{Nothing, Bool}=nothing) where K
    operator = operator
    lattice = latt
    amplitudes, indices = pair_amplitude_indices(latt, neighbors, a, b; amplitude=amplitude, intralayer=intralayer)
    return PairCorrelation{K}(operator, lattice, amplitudes, indices)
end

"""
    pair_amplitude_indices(latt::CustomLattice, neighbors::Neighbors, a::Integer, b::Integer; amplitude::Union{Nothing, Function}=nothing, intralayer::Union{Nothing, Bool}=nothing)
"""
function pair_amplitude_indices(latt::CustomLattice, neighbors::Neighbors, a::Integer, b::Integer; amplitude::Union{Nothing, Function}=nothing, intralayer::Union{Nothing, Bool}=nothing)
    bs = isnothing(intralayer) ? bonds(latt.lattice, neighbors) : intralayer ? filter(bond->bond.points[1].rcoordinate[3] == bond.points[2].rcoordinate[3], bonds(latt.lattice, neighbors)) : filter(bond->bond.points[1].rcoordinate[3] !== bond.points[2].rcoordinate[3], bonds(latt.lattice, neighbors))
    amp = Vector{Vector}(undef, length(latt.lattice))
    indices = Vector{Vector{Vector{Int}}}(undef, length(latt.lattice))
    for i in 1:length(latt.lattice)
        ibs = filter(bo -> any(p -> p.site == i, bo.points), bs)
        pos = [sort(collect(p.site for p in bo.points)) for bo in ibs]
        amp[i] = isnothing(amplitude) ? [1.0 for _ in 1:length(ibs)] : [amplitude(b) for b in ibs]
        if a == b
            indices[i] = map(p -> map(s -> latt.indices[s][a], p), pos)
        else
            indices[i] = [latt.indices[p][a], latt.indices[p][b] for p in pos]
        end
    end
    return amp, indices
end

"""
    correlator(correlation::PairCorrelation, gs::AbstractFiniteMPS; is=Vector((length(correlation.lattice.lattice)÷2):-1:1), js=Vector((length(correlation.lattice.lattice)÷2+1):1:length(correlation.lattice.lattice)))
"""
function correlator(correlation::PairCorrelation, gs::AbstractFiniteMPS; parallel::Union{String, Integer}=Threads.nthreads(), is=Vector((length(correlation.lattice.lattice)÷2):-1:1), js=Vector((length(correlation.lattice.lattice)÷2+1):1:length(correlation.lattice.lattice)))
    @assert length(is) == length(js) "Length of is and js must be the same"
    O, amplitudes, indices = correlation.operator, correlation.amplitudes, correlation.indices
    if parallel == "np"
        Fr = SharedArray{ComplexF64, 1}(length(is))
        @sync @distributed for i in eachindex(is) 
            Fr[i] = sum(correlator(O, gs, Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) * dot(amplitudes[is[i]][a], amplitudes[js[i]][b]) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
        end
    elseif parallel == 1
        Fr = zeros(Float64, length(is))
        for i in eachindex(is) 
            Fr[i] = sum(correlator(O, gs, Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) * dot(amplitudes[is[i]][a], amplitudes[js[i]][b]) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
        end
    else
        Fr = zeros(Float64, length(is))
        idx = Threads.Atomic{Int}(1)
        n = length(is)
        Threads.@sync for _ in 1:parallel
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1) 
                i > n && break  
                Fr[i] = sum(correlator(O, gs, Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) * dot(amplitudes[is[i]][a], amplitudes[js[i]][b]) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
            end
        end
    end
    return Fr
end

"""
    SpinCorrelation{K} <: AbstractCorrelation
"""
struct SpinCorrelation{K} <: AbstractCorrelation
    operator::AbstractTensorMap
    lattice::CustomLattice
    indices::AbstractArray{<:AbstractArray{<:Integer, 1}, 1}
end

"""
    spin_indices(latt::CustomLattice; a::Union{Nothing, Integer})
"""
function spin_indices(latt::CustomLattice; a::Union{Nothing, Integer})
    indices = isnothing(a) ? latt.indices : [[latt.indices[i][a],] for i in 1:length(latt.lattice)]
    return indices
end

"""
    SpinCorrelation{K}(operator::AbstractTensorMap, latt::CustomLattice, orbital::Integer)
"""
function SpinCorrelation{K}(operator::AbstractTensorMap, latt::CustomLattice, orbital::Integer) where K
    operator = operator
    lattice = latt
    indices = spin_indices(latt; a=orbital)
    return SpinCorrelation{K}(operator, lattice, amplitudes, indices)
end

"""
    correlator(correlation::SpinCorrelation, gs::AbstractFiniteMPS; is=Vector((length(correlation.lattice.lattice)÷2):-1:1), js=Vector((length(correlation.lattice.lattice)÷2+1):1:length(correlation.lattice.lattice)))
"""
function correlator(correlation::SpinCorrelation, gs::AbstractFiniteMPS; parallel::Union{String, Integer}=Threads.nthreads(), is=Vector((length(correlation.lattice.lattice)÷2):-1:1), js=Vector((length(correlation.lattice.lattice)÷2+1):1:length(correlation.lattice.lattice)))
    @assert length(is) == length(js) "Length of is and js must be the same"
    O, indices = correlation.operator, correlation.indices
    if parallel == "np"
        Fr = SharedArray{ComplexF64, 1}(length(is))
        @sync @distributed for i in eachindex(is) 
            Fr[i] = sum(correlator(O, gs, Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
        end
    elseif parallel == 1
        Fr = zeros(Float64, length(is))
        for i in eachindex(is) 
            Fr[i] = sum(correlator(O, gs, Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
        end
    else
        Fr = zeros(Float64, length(is))
        idx = Threads.Atomic{Int}(1)
        n = length(is)
        Threads.@sync for _ in 1:parallel
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1) 
                i > n && break  
                Fr[i] = sum(correlator(O, gs, Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
            end
        end
    end
    return Fr
end

"""
    correlator(O::AbstractTensorMap, gs::AbstractFiniteMPS, i::NTuple{1, Integer}, j::NTuple{1, Integer})
"""
function correlator(O::AbstractTensorMap, gs::AbstractFiniteMPS, i::NTuple{1, Integer}, j::NTuple{1, Integer})
    dot(chargedMPS(O, gs, i[1]), chargedMPS(O, gs, j[1]))
end

"""
    correlator(O::AbstractTensorMap, gs::AbstractFiniteMPS, i::NTuple{2, Integer}, j::NTuple{1, Integer})
"""
function correlator(O::AbstractTensorMap, gs::AbstractFiniteMPS, i::NTuple{2, Integer}, j::NTuple{2, Integer})
    dot(chargedMPS(O, gs, i[1], i[2]), chargedMPS(O, gs, j[1], j[2]))
end
