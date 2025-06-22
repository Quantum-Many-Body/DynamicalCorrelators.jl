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
        amp[i] = isnothing(amplitude) ? [1.0 for _ in 1:length(ibs)] : [amplitude(bo) for bo in ibs]
        if a == b
            indices[i] = map(p -> map(s -> latt.indices[s][a], p), pos)
        else
            indices[i] = map(p -> [latt.indices[p[1]][a], latt.indices[p[2]][b]], pos)
        end
    end
    return amp, indices
end

"""
    correlator(correlation::PairCorrelation, gs::AbstractFiniteMPS; is=Vector((length(correlation.lattice.lattice)÷2):-1:1), js=Vector((length(correlation.lattice.lattice)÷2+1):1:length(correlation.lattice.lattice)))
"""
function correlator(correlation::PairCorrelation, gs::AbstractFiniteMPS; 
                    parallel::Union{String, Integer}=Threads.nthreads(), 
                    is=Vector((length(correlation.lattice.lattice)÷2):-1:1), 
                    js=Vector((length(correlation.lattice.lattice)÷2+1):1:length(correlation.lattice.lattice)))
    @assert length(is) == length(js) "Length of is and js must be the same"
    O, amplitudes, indices = correlation.operator, correlation.amplitudes, correlation.indices
    if parallel == "np"
        Fr = SharedArray{Float64, 1}(length(is))
        @sync @distributed for i in eachindex(is) 
            Fr[i] = sum(correlator(O, gs, Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) * dot(amplitudes[is[i]][a], amplitudes[js[i]][b]) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
        end
        Fr = abs.(Fr[1:end])
    elseif parallel == 1
        Fr = zeros(Float64, length(is))
        for i in eachindex(is) 
            Fr[i] = abs(sum(correlator(O, gs, Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) * dot(amplitudes[is[i]][a], amplitudes[js[i]][b]) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]])))
        end
    else
        Fr = zeros(Float64, length(is))
        idx = Threads.Atomic{Int}(1)
        n = length(is)
        Threads.@sync for _ in 1:parallel
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1) 
                i > n && break  
                Fr[i] = abs(sum(correlator(O, gs, Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) * dot(amplitudes[is[i]][a], amplitudes[js[i]][b]) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]])))
            end
        end
    end
    return Fr
end

"""
    SiteCorrelation{K} <: AbstractCorrelation
"""
struct SiteCorrelation{K} <: AbstractCorrelation
    operator::AbstractTensorMap
    lattice::CustomLattice
    indices::AbstractArray{<:AbstractArray{<:Integer, 1}, 1}
end

"""
    site_indices(latt::CustomLattice; a::Union{Nothing, Integer})
"""
function site_indices(latt::CustomLattice; a::Union{Nothing, Integer})
    indices = isnothing(a) ? latt.indices : [[latt.indices[i][a],] for i in 1:length(latt.lattice)]
    return indices
end

"""
    SiteCorrelation{K}(operator::AbstractTensorMap, latt::CustomLattice, orbital::Integer)
"""
function SiteCorrelation{K}(operator::AbstractTensorMap, latt::CustomLattice, orbital::Integer) where K
    operator = operator
    lattice = latt
    indices = site_indices(latt; a=orbital)
    return SiteCorrelation{K}(operator, lattice, indices)
end

"""
    correlator(correlation::SiteCorrelation, gs::AbstractFiniteMPS; is=Vector((length(correlation.lattice.lattice)÷2):-1:1), js=Vector((length(correlation.lattice.lattice)÷2+1):1:length(correlation.lattice.lattice)))
"""
function correlator(correlation::SiteCorrelation, gs::AbstractFiniteMPS; 
                    single::Bool=false,
                    parallel::Union{String, Integer}=Threads.nthreads(), 
                    is=Vector((length(correlation.lattice.lattice)÷2):-1:1), 
                    js=Vector((length(correlation.lattice.lattice)÷2+1):1:length(correlation.lattice.lattice)))
    @assert length(is) == length(js) "Length of is and js must be the same"
    O, indices = correlation.operator, correlation.indices
    if parallel == "np"
        Fr = SharedArray{Float64, 1}(length(is))
        @sync @distributed for i in eachindex(is) 
            Fr[i] = single ? sum(dot(gs, chargedMPS(O, gs, indices[is[i]][a])) for a in 1:length(indices[is[i]])) : sum(correlator(O, gs, Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
        end
    elseif parallel == 1
        Fr = zeros(Float64, length(is))
        for i in eachindex(is) 
            Fr[i] = single ? sum(dot(gs, chargedMPS(O, gs, indices[is[i]][a])) for a in 1:length(indices[is[i]])) : sum(correlator(O, gs, Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
        end
    else
        Fr = zeros(Float64, length(is))
        idx = Threads.Atomic{Int}(1)
        n = length(is)
        Threads.@sync for _ in 1:parallel
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1) 
                i > n && break  
                Fr[i] = single ? sum(dot(gs, chargedMPS(O, gs, indices[is[i]][a])) for a in 1:length(indices[is[i]])) : sum(correlator(O, gs, Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
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

function correlator(state::AbstractFiniteMPS, O₁::AbstractTensorMap, O₂::AbstractTensorMap, i::Integer, j::Integer)
    i <= j || @error "i should be equal or smaller than j ($i, $j)"
    if i == j 
        O = contract_onesite(O₁, O₂)
        G = @plansor state.AC[i][1 2; 3] * O[4; 2] * conj(state.AC[i][1 4; 3])
    elseif  (length(domain(O₁)) == 2)&&(length(codomain(O₂)) == 2)
        @plansor Vₗ[-1 -2; -3] := state.AC[i][3 4; -3] * O₁[2; 4 -2] * conj(state.AC[i][3 2; -1])
        ctr = i + 1
        if j > ctr
            Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(j - 1)])
        end
        G = @plansor Vₗ[2 3; 5] * state.AR[j][5 6; 7] * O₂[3 4; 6] * conj(state.AR[j][2 4; 7])
    else
        @plansor Vₗ[-1; -3] := state.AC[i][3 4; -3] * O₁[2; 4] * conj(state.AC[i][3 2; -1])
        ctr = i + 1
        if j > ctr
            Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(j - 1)])
        end
        G = @plansor Vₗ[2; 5] * state.AR[j][5 6; 7] * O₂[4; 6] * conj(state.AR[j][2 4; 7])
    end
    return G
end


function correlator(state::AbstractFiniteMPS, O₁::AbstractTensorMap, O₂::AbstractTensorMap, i::Integer, j::Integer, k::Integer, l::Integer)
        O₁, O₂ = add_single_util_leg(O₁),  add_single_util_leg(O₂)
        I, J = decompose_localmpo(O₁)
        K, L = decompose_localmpo(O₂)
        U = ones(scalartype(state), _firstspace(O₁))
    if i < j < k < l
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if j > (i + 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[(i + 1):(j - 1)])
        end
        @plansor Vⱼ[-1 -2; -3] := Vᵢ[1 2; 3] * state.AR[j][3 4; -3] * J[2 5; 4 -2] *
                            conj(state.AR[j][1 5; -1])
        if k > (j + 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[(j + 1):(k - 1)])
        end
        @plansor Vₖ[-1 -2; -3] := Vⱼ[1 2; 3] * state.AR[k][3 4; -3] * K[2 5; 4 -2] *
                            conj(state.AR[k][1 5; -1])
        if l > (k + 1)
            Vₖ = Vₖ * TransferMatrix(state.AR[(k + 1):(l - 1)])
        end
        G = @plansor Vₖ[2 3; 5] * state.AR[l][5 6; 7] * L[3 4; 6 1] * U[1] *
                        conj(state.AR[l][2 4; 7])

    elseif i < j == k < l
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if j > (i + 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[(i + 1):(j - 1)])
        end
        @plansor Vⱼ[-1 -2; -3] := Vᵢ[1 2; 3] * state.AR[j][3 4; -3] * K[5 6; 4 -2] *
                            τ[7 8; 5 6] * J[2 9; 7 8] * conj(state.AR[j][1 9; -1])
        if l > (j + 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[(j + 1):(l - 1)])
        end
        G = @plansor Vⱼ[2 3; 5] * state.AR[l][5 6; 7] * L[3 4; 6 1] * U[1] *
                            conj(state.AR[l][2 4; 7])

    elseif i < k < j < l
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if k > (i + 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[(i + 1):(k - 1)])
        end                     
        @plansor Vₖ[-1 -2 -3 -4; -5] := Vᵢ[1 2; 3] * state.AR[k][3 4; -5] * 
                            K[5 6; 4 -4] * τ[5 7; 6 -3] * τ[2 8; 7 -2] * conj(state.AR[k][1 8; -1])
        if j > (k + 1)
             Vₖ = Vₖ * TransferMatrix(state.AR[(k + 1):(j - 1)])
        end
        @plansor Vⱼ[-1, -2; -3] := Vₖ[1 2 3 4; 5] * state.AR[j][5 6; -3] * τ[4 7; 6 -2] * 
                            τ[3 8; 7 9] * J[2 10; 8 9] * conj(state.AR[j][1 10; -1])
        if l > (j + 1)
             Vⱼ = Vⱼ * TransferMatrix(state.AR[(j + 1):(l - 1)])
        end
        G = @plansor Vⱼ[2 3; 5] * state.AR[l][5 6; 7] * L[3 4; 6 1] * U[1] *
                            conj(state.AR[l][2 4; 7])
    
    elseif i < k < j == l
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if k > (i + 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[(i + 1):(k - 1)])
        end                     
        @plansor Vₖ[-1 -2 -3 -4; -5] := Vᵢ[1 2; 3] * state.AR[k][3 4; -5] * 
                            K[5 6; 4 -4] * τ[5 7; 6 -3] * τ[2 8; 7 -2] * conj(state.AR[k][1 8; -1])
        if j > (k + 1)
             Vₖ = Vₖ * TransferMatrix(state.AR[(k + 1):(j - 1)])
        end
        G = @plansor Vₖ[1 2 3 4; 5] * state.AR[j][5 6; 11] * L[4 7; 6 12] * U[12] *
                            τ[3 8; 7 9] * J[2 10; 8 9] * conj(state.AR[j][1 10; 11])

    elseif i == k < j < l
        @plansor Vᵢ[-1 -2 -3 -4; -5] := state.AC[i][3 7; -5] * K[5 6; 7 -4] * τ[5 4; 6 -3] *
                            conj(U[1]) * I[1 2; 4 -2] * conj(state.AC[i][3 2; -1])
        if j > (i + 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[(i + 1):(j - 1)])
        end
        @plansor Vⱼ[-1 -2; -3] := Vᵢ[1 2 3 4; 5] * state.AR[j][5 6; -3] * τ[4 7; 6 -2] *
                            τ[3 8; 7 9] * J[2 10; 8 9] * conj(state.AR[j][1 10; -1])
        if l > (j + 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[(j + 1):(l - 1)])
        end
        G = @plansor Vⱼ[2 3; 5] * state.AR[l][5 6; 7] * L[3 4; 6 1] * U[1] *
                            conj(state.AR[l][2 4; 7])
    
    elseif i == k < j == l
        @plansor Vᵢ[-1 -2 -3 -4; -5] := state.AC[i][3 7; -5] * K[5 6; 7 -4] * τ[5 4; 6 -3] *
                            conj(U[1]) * I[1 2; 4 -2] * conj(state.AC[i][3 2; -1])
        if j > (i + 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[(i + 1):(j - 1)])
        end
        G = @plansor Vᵢ[1 2 3 4; 5] * state.AR[j][5 6; 11] * L[4 7; 6 12] * U[12] * 
                            τ[3 8; 7 9] * J[2 10; 8 9] * conj(state.AR[j][1 10; 11])
    
    elseif i < k < l < j
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if k > (i + 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[(i + 1):(k - 1)])
        end                     
        @plansor Vₖ[-1 -2 -3 -4; -5] := Vᵢ[1 2; 3] * state.AR[k][3 4; -5] * 
                            K[5 6; 4 -4] * τ[5 7; 6 -3] * τ[2 8; 7 -2] * conj(state.AR[k][1 8; -1])
        if l > (k + 1)
            Vₖ = Vₖ * TransferMatrix(state.AR[(k + 1):(l - 1)])
        end
        @plansor Vₗ[-1 -2 -3; -4] := Vₖ[1 2 3 4; 5] * state.AR[l][5 6; -4] * L[4 7; 6 10] * U[10] *
                            τ[3 8; 7 -3] * τ[2 9; 8 -2] * conj(state.AR[l][1 9; -1])
        if j > (l + 1)
            Vₗ = Vₗ * TransferMatrix(state.AR[(l + 1):(j - 1)])
        end
        G = @plansor Vₗ[1 2 3; 4] * state.AR[j][4 5; 9] *
                            τ[3 6; 5 7] * J[2 8; 6 7] * conj(state.AR[j][1 8; 9])

    elseif i == k < l < j
        @plansor Vᵢ[-1 -2 -3 -4; -5] := state.AC[i][3 7; -5] * K[5 6; 7 -4] * τ[5 4; 6 -3] *
                            conj(U[1]) * I[1 2; 4 -2] * conj(state.AC[i][3 2; -1])
        if l > (i + 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[(i + 1):(l - 1)])
        end
        @plansor Vₗ[-1 -2 -3; -4] := Vᵢ[1 2 3 4; 5] * state.AR[l][5 6; -4] * L[4 7; 6 10] * U[10] *
                            τ[3 8; 7 -3] * τ[2 9; 8 -2] * conj(state.AR[l][1 9; -1])
        if j > (l + 1)
            Vₗ = Vₗ * TransferMatrix(state.AR[(l + 1):(j - 1)])
        end
        G = @plansor Vₗ[1 2 3; 4] * state.AR[j][4 5; 9] *
                            τ[3 6; 5 7] * J[2 8; 6 7] * conj(state.AR[j][1 8; 9])

    elseif k < i < j < l
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] *
                            τ[3 5; 4 -2] * conj(state.AC[k][1 5; -1])
        if i > (k + 1)
            Vₖ = Vₖ * TransferMatrix(state.AR[(k + 1):(i - 1)])
        end
        @plansor Vᵢ[-1 -2 -3 -4; -5] := Vₖ[1 2 3; 4] * state.AR[i][4 5; -5] * τ[3 6; 5 -4] * 
                            τ[2 8; 6 -3] * conj(U[7]) * I[7 9; 8 -2] * conj(state.AR[i][1 9; -1])

        if j > (i + 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[(i + 1):(j - 1)])
        end
        @plansor Vⱼ[-1 -2; -3] := Vᵢ[1 2 3 4; 5] * state.AR[j][5 6; -3] * τ[4 7; 6 -2] * τ[3 8; 7 9] *
                            J[2 10; 8 9] * conj(state.AR[j][1 10; -1])
        if l > (j + 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[(j + 1):(l - 1)])
        end
        G = @plansor Vⱼ[1 2; 3] * state.AR[l][3 4; 6] * L[2 5; 4 7] * U[7] * conj(state.AR[l][1 5; 6])

    elseif k < i < j == l
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] *
                            τ[3 5; 4 -2] * conj(state.AC[k][1 5; -1])
        if i > (k + 1)
            Vₖ = Vₖ * TransferMatrix(state.AR[(k + 1):(i - 1)])
        end
        @plansor Vᵢ[-1 -2 -3 -4; -5] := Vₖ[1 2 3; 4] * state.AR[i][4 5; -5] * τ[3 6; 5 -4] * 
                            τ[2 8; 6 -3] * conj(U[7]) * I[7 9; 8 -2] * conj(state.AR[i][1 9; -1])

        if j > (i + 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[(i + 1):(j - 1)])
        end
        G = @plansor Vᵢ[1 2 3 4; 5] * state.AR[j][5 6; 12] * L[4 8; 6 7] * U[7] * τ[3 9; 8 10] *
                            J[2 11; 9 10] * conj(state.AR[j][1 11; 12])
        
    elseif k < i < l < j
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] *
                            τ[3 5; 4 -2] * conj(state.AC[k][1 5; -1])
        if i > (k + 1)
            Vₖ = Vₖ * TransferMatrix(state.AR[(k + 1):(i - 1)])
        end
        @plansor Vᵢ[-1 -2 -3 -4; -5] := Vₖ[1 2 3; 4] * state.AR[i][4 5; -5] * τ[3 6; 5 -4] * 
                            τ[2 8; 6 -3] * conj(U[7]) * I[7 9; 8 -2] * conj(state.AR[i][1 9; -1])
        if l > (i + 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[(i + 1):(l - 1)])
        end
        @plansor Vₗ[-1 -2 -3; -4] := Vᵢ[1 2 3 4; 5] * state.AR[l][5 6; -4] * L[4 8; 6 7] * U[7] *
                            τ[3 9; 8 -3] * τ[2 10; 9 -2] * conj(state.AR[l][1 10; -1])
        if j > (l + 1)
            Vₗ = Vₗ * TransferMatrix(state.AR[(l + 1):(j - 1)])
        end
        G = @plansor Vₗ[1 2 3; 4] * state.AR[j][4 5; 9] * τ[3 6; 5 7] * J[2 8; 6 7] * 
                            conj(state.AR[j][1 8; 9])

    elseif k < i == l < j
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] *
                            τ[3 5; 4 -2] * conj(state.AC[k][1 5; -1])
        if i > (k + 1)
            Vₖ = Vₖ * TransferMatrix(state.AR[(k + 1):(i - 1)])
        end
        @plansor Vᵢ[-1 -2 -3; -4] := Vₖ[1 2 3; 4] * state.AR[i][4 5; -4] * L[3 7; 5 6] * U[6] *
                            τ[2 9; 7 -3] * I[8 10; 9 -2] * conj(U[8]) * conj(state.AR[i][1 10; -1])
        if j > (i + 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[(i + 1):(j - 1)])
        end
        G = @plansor Vᵢ[1 2 3; 4] * state.AR[j][4 5; 6] *
                            τ[3 7; 5 8] * J[2 9; 7 8] * conj(state.AR[j][1 9; 6])

    elseif k < l < i < j
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] *
                            τ[3 5; 4 -2] * conj(state.AC[k][1 5; -1])
        if l > (k + 1)
            Vₖ = Vₖ * TransferMatrix(state.AR[(k + 1):(l - 1)])
        end
        @plansor Vₗ[-1 -2; -3] := Vₖ[1 2 3; 4] * state.AR[l][4 6; -3] * L[3 7; 6 5] * 
                            U[5] * τ[2 8; 7 -2] * conj(state.AR[l][1 8; -1])
        if i > (l + 1)
                Vₗ = Vₗ * TransferMatrix(state.AR[(l + 1):(i - 1)])
        end
        @plansor Vᵢ[-1 -2 -3; -4] := Vₗ[1 2; 3] * state.AR[i][3 4; -4] * τ[2 6; 4 -3] * 
                            I[5 7; 6 -2] * conj(U[5]) * conj(state.AR[i][1 7; -1])
        if j > (i + 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[(i + 1):(j - 1)])
        end
        G = @plansor Vᵢ[1 2 3; 4] * state.AR[j][4 5; 9] * τ[3 6; 5 7] * J[2 8; 6 7] * 
                            conj(state.AR[j][1 8; 9]) 
    else
        throw(ArgumentError("invalid input indices (i, j, k, l) for ($i, $j, $k, $l), only (i < j) && (k < l) is valid"))
    end
    return G
end
