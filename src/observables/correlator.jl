"""
    abstract type AbstractCorrelation end
"""
abstract type AbstractCorrelation end

"""
    PairCorrelation <: AbstractCorrelation
"""
struct PairCorrelation <: AbstractCorrelation
    operators::Tuple{<:AbstractTensorMap, <:AbstractTensorMap}
    lattice::CustomLattice
    amplitudes::AbstractArray
    indices::AbstractArray
end

"""
    PairCorrelation(ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap}, latt::CustomLattice, neighbors::Neighbors, a::Integer, b::Integer; amplitude::Union{Nothing, Function}=nothing, intralayer::Union{Nothing, Bool}=nothing) 
"""
function PairCorrelation(ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap}, latt::CustomLattice, neighbors::Neighbors, a::Integer, b::Integer; amplitude::Union{Nothing, Function}=nothing, intralayer::Union{Nothing, Bool}=nothing)
    amplitudes, indices = pair_amplitude_indices(latt, neighbors, a, b; amplitude=amplitude, intralayer=intralayer)
    return PairCorrelation(ops, latt, amplitudes, indices)
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
    ops, amplitudes, indices = correlation.operators, correlation.amplitudes, correlation.indices
    if parallel == "np"
        Fr = SharedArray{Float64, 1}(length(is))
        @sync @distributed for i in eachindex(is) 
            Fr[i] = sum(correlator(gs, ops[1], ops[2], Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) * dot(amplitudes[is[i]][a], amplitudes[js[i]][b]) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
        end
        Fr = abs.(Fr[1:end])
    elseif parallel == 1
        Fr = zeros(Float64, length(is))
        for i in eachindex(is) 
            Fr[i] = sum(correlator(gs, ops[1], ops[2], Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) * dot(amplitudes[is[i]][a], amplitudes[js[i]][b]) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
        end
    else
        Fr = zeros(Float64, length(is))
        idx = Threads.Atomic{Int}(1)
        n = length(is)
        Threads.@sync for _ in 1:parallel
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1) 
                i > n && break  
                Fr[i] = sum(correlator(gs, ops[1], ops[2], Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) * dot(amplitudes[is[i]][a], amplitudes[js[i]][b]) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
            end
        end
    end
    return Fr
end

"""
    TwoSiteCorrelation <: AbstractCorrelation
"""
struct TwoSiteCorrelation <: AbstractCorrelation
    operators::Tuple{<:AbstractTensorMap, <:AbstractTensorMap}
    lattice::CustomLattice
    indices::AbstractArray{<:AbstractArray{<:Integer, 1}, 1}
end

"""
    OneSiteCorrelation <: AbstractCorrelation
"""
struct OneSiteCorrelation <: AbstractCorrelation
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
    TwoSiteCorrelation(operators::Tuple{<:AbstractTensorMap, <:AbstractTensorMap}, latt::CustomLattice, orbital::Integer)
"""
function TwoSiteCorrelation(operators::Tuple{<:AbstractTensorMap, <:AbstractTensorMap}, latt::CustomLattice, orbital::Integer)
    indices = site_indices(latt; a=orbital)
    return TwoSiteCorrelation(operators, latt, indices)
end

"""
    OneSiteCorrelation{K}(operator::AbstractTensorMap, latt::CustomLattice, orbital::Integer)
"""
function OneSiteCorrelation(operator::AbstractTensorMap, latt::CustomLattice, orbital::Integer)
    indices = site_indices(latt; a=orbital)
    return OneSiteCorrelation(operator, latt, indices)
end

"""
    correlator(correlation::TwoSiteCorrelation, gs::AbstractFiniteMPS; is=Vector((length(correlation.lattice.lattice)÷2):-1:1), js=Vector((length(correlation.lattice.lattice)÷2+1):1:length(correlation.lattice.lattice)))
"""
function correlator(correlation::TwoSiteCorrelation, gs::AbstractFiniteMPS; 
                    parallel::Union{String, Integer}=Threads.nthreads(), 
                    is=Vector((length(correlation.lattice.lattice)÷2):-1:1), 
                    js=Vector((length(correlation.lattice.lattice)÷2+1):1:length(correlation.lattice.lattice)))
    @assert length(is) == length(js) "Length of is and js must be the same"
    ops, indices = correlation.operators, correlation.indices
    if parallel == "np"
        Fr = SharedArray{Float64, 1}(length(is))
        @sync @distributed for i in eachindex(is) 
            Fr[i] = sum(correlator(gs, ops[1], ops[2], Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
        end
    elseif parallel == 1
        Fr = zeros(Float64, length(is))
        for i in eachindex(is) 
            Fr[i] = sum(correlator(gs, ops[1], ops[2], Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
        end
    else
        Fr = zeros(Float64, length(is))
        idx = Threads.Atomic{Int}(1)
        n = length(is)
        Threads.@sync for _ in 1:parallel
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1) 
                i > n && break  
                Fr[i] = sum(correlator(gs, ops[1], ops[2], Tuple(indices[is[i]][a]), Tuple(indices[js[i]][b])) for a in 1:length(indices[is[i]]), b in 1:length(indices[js[i]]))
            end
        end
    end
    return Fr
end

"""
    correlator(correlation::OneSiteCorrelation, gs::AbstractFiniteMPS; is=Vector((length(correlation.lattice.lattice)÷2):-1:1), js=Vector((length(correlation.lattice.lattice)÷2+1):1:length(correlation.lattice.lattice)))
"""
function correlator(correlation::OneSiteCorrelation, gs::AbstractFiniteMPS; 
                    parallel::Union{String, Integer}=Threads.nthreads(), 
                    is=Vector((length(correlation.lattice.lattice)÷2):-1:1), 
                    js=Vector((length(correlation.lattice.lattice)÷2+1):1:length(correlation.lattice.lattice)))
    @assert length(is) == length(js) "Length of is and js must be the same"
    O, indices = correlation.operator, correlation.indices
    if parallel == "np"
        Fr = SharedArray{Float64, 1}(length(is))
        @sync @distributed for i in eachindex(is) 
            Fr[i] = sum(correlator(gs, O, indices[is[i]][a]) for a in 1:length(indices[is[i]]))
        end
    elseif parallel == 1
        Fr = zeros(Float64, length(is))
        for i in eachindex(is) 
            Fr[i] = sum(correlator(gs, O, indices[is[i]][a]) for a in 1:length(indices[is[i]]))
        end
    else
        Fr = zeros(Float64, length(is))
        idx = Threads.Atomic{Int}(1)
        n = length(is)
        Threads.@sync for _ in 1:parallel
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1) 
                i > n && break  
                Fr[i] = sum(correlator(gs, O, indices[is[i]][a]) for a in 1:length(indices[is[i]]))
            end
        end
    end
    return Fr
end

function correlator(gs::AbstractFiniteMPS, O₁::AbstractTensorMap, O₂::AbstractTensorMap; is=1:length(gs), js=1:length(gs), parallel=Threads.nthreads())
    Fr = zeros(ComplexF64, length(is), length(js))
    idx = Threads.Atomic{Int}(1)
    indices = CartesianIndices((length(is), length(js)))
    n = length(indices)
    Threads.@sync for _ in 1:parallel
        Threads.@spawn while true
            i = Threads.atomic_add!(idx, 1) 
            i > n && break  
            a, b = indices[i].I
            Fr[a, b] = correlator(gs, O₁, O₂, (is[a],), (js[b],))
        end
    end
    return Fr
end

"""
    correlator(state::AbstractFiniteMPS, O::AbstractTensorMap, i::Integer)
"""
function correlator(state::AbstractFiniteMPS, O::AbstractTensorMap, i::Integer)
    G = @plansor state.AC[i][1 2; 4] * O[3; 2] * conj(state.AC[i][1 3; 4])
    return G
end

"""
    correlator(state::AbstractFiniteMPS, O₁::AbstractTensorMap, O₂::AbstractTensorMap, i::NTuple{1, Integer}, j::NTuple{1, Integer})
"""
function correlator(state::AbstractFiniteMPS, O₁::AbstractTensorMap, O₂::AbstractTensorMap, i::NTuple{1, Integer}, j::NTuple{1, Integer})
    i, j = i[1], j[1]
    if (length(domain(O₁)) == 2)&&(length(codomain(O₂)) == 2)
        if i == j 
            O = contract_onesite(O₁, O₂)
            G = @plansor state.AC[i][1 2; 3] * O[4; 2] * conj(state.AC[i][1 4; 3])
        elseif  i < j
            @plansor Vₗ[-1 -2; -3] := state.AC[i][3 4; -3] * O₁[2; 4 -2] * conj(state.AC[i][3 2; -1])
            ctr = i + 1
            if j > ctr
                Z = TensorMap(BraidingTensor(domain(O₁,1), domain(O₁,2)))
                midsites = ctr:(j - 1)
                Vₗ = Vₗ * TransferMatrix(state.AR[midsites], fill(Z, length(midsites)), state.AR[midsites])
            end
            G = @plansor Vₗ[2 3; 5] * state.AR[j][5 6; 7] * O₂[3 4; 6] * conj(state.AR[j][2 4; 7])
        else
            iso₂ = isomorphism(flip(codomain(O₂, 1)), codomain(O₂, 1))
            @plansor Vₗ[-1 -2; -3] := state.AC[j][3 4; -3] * O₂[2 1; 4] * iso₂[6; 2] * τ[6 5; 1 -2] * conj(state.AC[j][3 5; -1])
            ctr = j + 1
            if i > ctr
                Z = TensorMap(BraidingTensor(domain(O₁,1), dual(flip(codomain(O₂, 1)))))
                midsites = ctr:(i - 1)
                Vₗ = Vₗ * TransferMatrix(state.AR[midsites], fill(Z, length(midsites)), state.AR[midsites])
            end
            iso₁ = isomorphism(codomain(O₂, 1), flip(codomain(O₂, 1)))
            G = @plansor Vₗ[1 2; 3] * state.AR[i][3 4; 8] * τ[2 5; 4 6] * iso₁[9; 6] * O₁[7; 5 9] * conj(state.AR[i][1 7; 8])
        end
    elseif (length(domain(O₁)) == 1)&&(length(codomain(O₂)) == 1)
        if i == j 
        O = contract_onesite(O₁, O₂)
        G = @plansor state.AC[i][1 2; 3] * O[4; 2] * conj(state.AC[i][1 4; 3])
        elseif i < j
            @plansor Vₗ[-1; -3] := state.AC[i][3 4; -3] * O₁[2; 4] * conj(state.AC[i][3 2; -1])
            ctr = i + 1
            if j > ctr
                Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(j - 1)])
            end
            G = @plansor Vₗ[2; 5] * state.AR[j][5 6; 7] * O₂[4; 6] * conj(state.AR[j][2 4; 7])
        else
            @plansor Vⱼ[-1; -3] := state.AC[j][3 4; -3] * O₂[2; 4] * conj(state.AC[j][3 2; -1])
            ctr = j + 1
            if i > ctr
                Vⱼ = Vⱼ * TransferMatrix(state.AR[ctr:(i - 1)])
            end
            G = @plansor Vⱼ[2; 5] * state.AR[i][5 6; 7] * O₁[4; 6] * conj(state.AR[i][2 4; 7])
        end
    else
        throw(ArgumentError("invalid legs of O₁ and O₂!"))
    end
    return G
end

"""
    correlator(state::AbstractFiniteMPS, O₁::AbstractTensorMap, O₂::AbstractTensorMap, ij::NTuple{2, Integer}, kl::NTuple{2, Integer})
"""
function correlator(state::AbstractFiniteMPS, O₁::AbstractTensorMap, O₂::AbstractTensorMap, ij::NTuple{2, Integer}, kl::NTuple{2, Integer})
        O₁, O₂ = add_single_util_leg(O₁),  add_single_util_leg(O₂)
        I, J = decompose_localmpo(O₁)
        K, L = decompose_localmpo(O₂)
        U = ones(scalartype(state), _firstspace(O₁))
        i, j = ij
        k, l = kl
    if i < j < k < l
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if j > (i + 1)
            Zᵢⱼ = TensorMap(BraidingTensor(domain(I,1), domain(I, 2)))
            midsites = (i + 1):(j - 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[midsites], fill(Zᵢⱼ, length(midsites)), state.AR[midsites])
        end
        @plansor Vⱼ[-1 -2; -3] := Vᵢ[1 2; 3] * state.AR[j][3 4; -3] * J[2 5; 4 -2] *
                            conj(state.AR[j][1 5; -1])
        if k > (j + 1)
            Zⱼₖ = TensorMap(BraidingTensor(domain(J, 1), domain(J, 2)))
            midsites = (j + 1):(k - 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[midsites], fill(Zⱼₖ, length(midsites)), state.AR[midsites])
        end
        @plansor Vₖ[-1 -2; -3] := Vⱼ[1 2; 3] * state.AR[k][3 4; -3] * K[2 5; 4 -2] *
                            conj(state.AR[k][1 5; -1])
        if l > (k + 1)
            Zₖₗ = TensorMap(BraidingTensor(domain(K, 1), domain(K, 2)))
            midsites = (k + 1):(l - 1)
            Vₖ = Vₖ * TransferMatrix(state.AR[midsites], fill(Zₖₗ, length(midsites)), state.AR[midsites])
        end
        G = @plansor Vₖ[2 3; 5] * state.AR[l][5 6; 7] * L[3 4; 6 1] * U[1] *
                        conj(state.AR[l][2 4; 7])

    elseif i < j == k < l
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if j > (i + 1)
            Zᵢⱼ = TensorMap(BraidingTensor(domain(I,1), domain(I, 2)))
            midsites = (i + 1):(j - 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[midsites], fill(Zᵢⱼ, length(midsites)), state.AR[midsites])
        end
        @plansor Vⱼ[-1 -2; -3] := Vᵢ[1 2; 3] * state.AR[j][3 4; -3] * K[5 6; 4 -2] *
                            τ[5 7; 6 8] * J[2 9; 7 8] * conj(state.AR[j][1 9; -1])
        if l > (j + 1)
            Zₖₗ = TensorMap(BraidingTensor(domain(K, 1), domain(K, 2)))
            midsites = (j + 1):(l - 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[midsites], fill(Zₖₗ, length(midsites)), state.AR[midsites])
        end
        G = @plansor Vⱼ[2 3; 5] * state.AR[l][5 6; 7] * L[3 4; 6 1] * U[1] *
                            conj(state.AR[l][2 4; 7])

    elseif i < k < j < l
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if k > (i + 1)
            Zᵢₖ = TensorMap(BraidingTensor(domain(I,1), domain(I, 2)))
            midsites = (i + 1):(k - 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[midsites], fill(Zᵢₖ, length(midsites)), state.AR[midsites])
        end                     
        iso = isomorphism(flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vₖ[-1 -2 -3 -4; -5] := Vᵢ[1 2; 3] * state.AR[k][3 4; -5] * 
                            K[5 6; 4 -4] * iso[9; 5] * τ[9 7; 6 -3] * τ[2 8; 7 -2] * conj(state.AR[k][1 8; -1])
        if j > (k + 1)
            τ1 = TensorMap(BraidingTensor(codomain(K, 2), domain(K, 2)))
            τ2 = TensorMap(BraidingTensor(codomain(K, 2), dual(flip(codomain(K, 1)))))
            τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
            for a in (k + 1):(j - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(codomain(K, 1), flip(codomain(K, 1)))
        @plansor Vⱼ[-1, -2; -3] := Vₖ[1 2 3 4; 5] * state.AR[j][5 6; -3] * τ[4 7; 6 -2] * 
                            τ[3 8; 7 9] * J[2 10; 8 11] * iso[11; 9] * conj(state.AR[j][1 10; -1])
        if l > (j + 1)
            Zⱼₗ = TensorMap(BraidingTensor(domain(K, 1), domain(K, 2)))
            midsites = (j + 1):(l - 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[midsites], fill(Zⱼₗ, length(midsites)), state.AR[midsites])
        end
        G = @plansor Vⱼ[2 3; 5] * state.AR[l][5 6; 7] * L[3 4; 6 1] * U[1] *
                            conj(state.AR[l][2 4; 7])
    
    elseif i < k < j == l
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if k > (i + 1)
            Zᵢₖ = TensorMap(BraidingTensor(domain(I,1), domain(I, 2)))
            midsites = (i + 1):(k - 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[midsites], fill(Zᵢₖ, length(midsites)), state.AR[midsites])
        end                     
        iso = isomorphism(flip(codomain(K, 1)), codomain(K, 1))
        τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
        @plansor Vₖ[-1 -2 -3 -4; -5] := Vᵢ[1 2; 3] * state.AR[k][3 4; -5] * 
                            K[5 6; 4 -4] * iso[9; 5] * τ[9 7; 6 -3] * τ3[2 8; 7 -2] * conj(state.AR[k][1 8; -1])
        if j > (k + 1)
            τ1 = TensorMap(BraidingTensor(codomain(K, 2), domain(K, 2)))
            τ2 = TensorMap(BraidingTensor(codomain(K, 2), dual(flip(codomain(K, 1)))))
            τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
            for a in (k + 1):(j - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vₖ[1 2 3 4; 5] * state.AR[j][5 6; 13] * L[4 7; 6 12] * U[12] *
                            τ[3 8; 7 9] * iso[11; 9] * J[2 10; 8 11] * conj(state.AR[j][1 10; 13])

    elseif i == k < j < l
        iso = isomorphism(flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vᵢ[-1 -2 -3 -4; -5] := state.AC[i][3 7; -5] * K[5 6; 7 -4] * iso[8; 5] * τ[8 4; 6 -3] *
                            conj(U[1]) * I[1 2; 4 -2] * conj(state.AC[i][3 2; -1])
        if j > (i + 1)
            τ1 = TensorMap(BraidingTensor(codomain(K, 2), domain(K, 2)))
            τ2 = TensorMap(BraidingTensor(codomain(K, 2), dual(flip(codomain(K, 1)))))
            τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(j - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(codomain(K, 1), flip(codomain(K, 1)))
        τ1 = TensorMap(BraidingTensor(codomain(K, 2), domain(K, 2)))
        @plansor Vⱼ[-1, -2; -3] := Vᵢ[1 2 3 4; 5] * state.AR[j][5 6; -3] * τ1[4 7; 6 -2] * 
                            τ[3 8; 7 9] * J[2 10; 8 11] * iso[11; 9] * conj(state.AR[j][1 10; -1])
        if l > (j + 1)
            Zⱼₗ = TensorMap(BraidingTensor(domain(K, 1), domain(K, 2)))
            midsites = (j + 1):(l - 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[midsites], fill(Zⱼₗ, length(midsites)), state.AR[midsites])
        end
        G = @plansor Vⱼ[2 3; 5] * state.AR[l][5 6; 7] * L[3 4; 6 1] * U[1] *
                            conj(state.AR[l][2 4; 7])
    
    elseif i == k < j == l
        iso = isomorphism(flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vᵢ[-1 -2 -3 -4; -5] := state.AC[i][3 7; -5] * K[5 6; 7 -4] * iso[8; 5] * τ[8 4; 6 -3] *
                            conj(U[1]) * I[1 2; 4 -2] * conj(state.AC[i][3 2; -1])
        if j > (i + 1)
            τ1 = TensorMap(BraidingTensor(codomain(K, 2), domain(K, 2)))
            τ2 = TensorMap(BraidingTensor(codomain(K, 2), dual(flip(codomain(K, 1)))))
            τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(j - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vᵢ[1 2 3 4; 5] * state.AR[j][5 6; 13] * L[4 7; 6 12] * U[12] *
                            τ[3 8; 7 9] * iso[11; 9] * J[2 10; 8 11] * conj(state.AR[j][1 10; 13])
    
    elseif i < k < l < j
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if k > (i + 1)
            Zᵢₖ = TensorMap(BraidingTensor(domain(I,1), domain(I, 2)))
            midsites = (i + 1):(k - 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[midsites], fill(Zᵢₖ, length(midsites)), state.AR[midsites])
        end                     
        iso = isomorphism(flip(codomain(K, 1)), codomain(K, 1))
        τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
        @plansor Vₖ[-1 -2 -3 -4; -5] := Vᵢ[1 2; 3] * state.AR[k][3 4; -5] * 
                            K[5 6; 4 -4] * iso[9; 5] * τ[9 7; 6 -3] * τ3[2 8; 7 -2] * conj(state.AR[k][1 8; -1])
        if l > (k + 1)
            τ1 = TensorMap(BraidingTensor(codomain(K, 2), domain(K, 2)))
            τ2 = TensorMap(BraidingTensor(codomain(K, 2), dual(flip(codomain(K, 1)))))
            τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
            for a in (k + 1):(l - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        τ2 = TensorMap(BraidingTensor(codomain(K, 2), dual(flip(codomain(K, 1)))))
        @plansor Vₗ[-1 -2 -3; -4] := Vₖ[1 2 3 4; 5] * state.AR[l][5 6; -4] * L[4 7; 6 10] * U[10] *
                            τ2[3 8; 7 -3] * τ3[2 9; 8 -2] * conj(state.AR[l][1 9; -1])
        if j > (l + 1)
            for a in (l + 1):(j - 1)
                Vₗ = transfer_left(Vₗ, state.AR[a], τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vₗ[1 2 3; 4] * state.AR[j][4 5; 9] *
                            τ[3 6; 5 7] * iso[10; 7] * J[2 8; 6 10] * conj(state.AR[j][1 8; 9])

    elseif i == k < l < j
        iso = isomorphism(flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vᵢ[-1 -2 -3 -4; -5] := state.AC[i][3 7; -5] * K[5 6; 7 -4] * iso[8; 5] * τ[8 4; 6 -3] *
                            conj(U[1]) * I[1 2; 4 -2] * conj(state.AC[i][3 2; -1])
        if l > (i + 1)
            τ1 = TensorMap(BraidingTensor(codomain(K, 2), domain(K, 2)))
            τ2 = TensorMap(BraidingTensor(codomain(K, 2), dual(flip(codomain(K, 1)))))
            τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(l - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        τ2 = TensorMap(BraidingTensor(codomain(K, 2), dual(flip(codomain(K, 1)))))
        τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
        @plansor Vₗ[-1 -2 -3; -4] := Vᵢ[1 2 3 4; 5] * state.AR[l][5 6; -4] * L[4 7; 6 10] * U[10] *
                            τ2[3 8; 7 -3] * τ3[2 9; 8 -2] * conj(state.AR[l][1 9; -1])
        if j > (l + 1)
            for a in (l + 1):(j - 1)
                Vₗ = transfer_left(Vₗ, state.AR[a], τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vₗ[1 2 3; 4] * state.AR[j][4 5; 9] *
                            τ[3 6; 5 7] * iso[10; 7] * J[2 8; 6 10] * conj(state.AR[j][1 8; 9])

    elseif k < i < j < l
        iso = isomorphism(flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] * iso[5; 3] * 
                            τ[5 6; 4 -2] * conj(state.AC[k][1 6; -1])
        τ1 = TensorMap(BraidingTensor(codomain(K, 2), domain(K, 2)))
        τ2 = TensorMap(BraidingTensor(codomain(K, 2), dual(flip(codomain(K, 1)))))
        if i > (k + 1)
            for a in (k + 1):(i - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, state.AR[a])
            end
        end
        @plansor Vᵢ[-1 -2 -3 -4; -5] := Vₖ[1 2 3; 4] * state.AR[i][4 5; -5] * τ1[3 6; 5 -4] * 
                            τ2[2 8; 6 -3] * conj(U[7]) * I[7 9; 8 -2] * conj(state.AR[i][1 9; -1])
        if j > (i + 1)
            τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(j - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(codomain(K, 1), flip(codomain(K, 1)))
        @plansor Vⱼ[-1 -2; -3] := Vᵢ[1 2 3 4; 5] * state.AR[j][5 6; -3] * τ1[4 7; 6 -2] * τ2[3 8; 7 9] *
                            J[2 10; 8 11] * iso[11; 9] * conj(state.AR[j][1 10; -1])
        if l > (j + 1)
            Zⱼₗ = TensorMap(BraidingTensor(domain(K, 1), domain(K, 2)))
            midsites = (j + 1):(l - 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[midsites], fill(Zⱼₗ, length(midsites)), state.AR[midsites])
        end
        G = @plansor Vⱼ[1 2; 3] * state.AR[l][3 4; 6] * L[2 5; 4 7] * U[7] * conj(state.AR[l][1 5; 6])

    elseif k < i < j == l
        iso = isomorphism(flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] * iso[5; 3] * 
                            τ[5 6; 4 -2] * conj(state.AC[k][1 6; -1])
        τ1 = TensorMap(BraidingTensor(codomain(K, 2), domain(K, 2)))
        τ2 = TensorMap(BraidingTensor(codomain(K, 2), dual(flip(codomain(K, 1)))))
        if i > (k + 1)
            for a in (k + 1):(i - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, state.AR[a])
            end
        end
        @plansor Vᵢ[-1 -2 -3 -4; -5] := Vₖ[1 2 3; 4] * state.AR[i][4 5; -5] * τ1[3 6; 5 -4] * 
                            τ2[2 8; 6 -3] * conj(U[7]) * I[7 9; 8 -2] * conj(state.AR[i][1 9; -1])
        if j > (i + 1)
            τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(j - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vᵢ[1 2 3 4; 5] * state.AR[j][5 6; 12] * L[4 8; 6 7] * U[7] * τ[3 9; 8 10] * iso[13; 10] *
                            J[2 11; 9 13] * conj(state.AR[j][1 11; 12])
        
    elseif k < i < l < j
        iso = isomorphism(flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] * iso[5; 3] * 
                            τ[5 6; 4 -2] * conj(state.AC[k][1 6; -1])
        τ1 = TensorMap(BraidingTensor(codomain(K, 2), domain(K, 2)))
        τ2 = TensorMap(BraidingTensor(codomain(K, 2), dual(flip(codomain(K, 1)))))
        if i > (k + 1)
            for a in (k + 1):(i - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, state.AR[a])
            end
        end
        @plansor Vᵢ[-1 -2 -3 -4; -5] := Vₖ[1 2 3; 4] * state.AR[i][4 5; -5] * τ1[3 6; 5 -4] * 
                            τ2[2 8; 6 -3] * conj(U[7]) * I[7 9; 8 -2] * conj(state.AR[i][1 9; -1])
        if l > (i + 1)
            τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(l - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
        @plansor Vₗ[-1 -2 -3; -4] := Vᵢ[1 2 3 4; 5] * state.AR[l][5 6; -4] * L[4 7; 6 10] * U[10] *
                            τ2[3 8; 7 -3] * τ3[2 9; 8 -2] * conj(state.AR[l][1 9; -1])
        if j > (l + 1)
            for a in (l + 1):(j - 1)
                Vₗ = transfer_left(Vₗ, state.AR[a], τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vₗ[1 2 3; 4] * state.AR[j][4 5; 9] *
                            τ[3 6; 5 7] * iso[10; 7] * J[2 8; 6 10] * conj(state.AR[j][1 8; 9])

    elseif k < i == l < j
        iso = isomorphism(flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] * iso[5; 3] * 
                            τ[5 6; 4 -2] * conj(state.AC[k][1 6; -1])
        τ1 = TensorMap(BraidingTensor(codomain(K, 2), domain(K, 2)))
        τ2 = TensorMap(BraidingTensor(codomain(K, 2), dual(flip(codomain(K, 1)))))
        if i > (k + 1)
            for a in (k + 1):(i - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, state.AR[a])
            end
        end
        @plansor Vᵢ[-1 -2 -3; -4] := Vₖ[1 2 3; 4] * state.AR[i][4 5; -4] * L[3 7; 5 6] * U[6] *
                            τ2[2 9; 7 -3] * I[8 10; 9 -2] * conj(U[8]) * conj(state.AR[i][1 10; -1])
        if j > (i + 1)
            τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(j - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vᵢ[1 2 3; 4] * state.AR[j][4 5; 9] *
                            τ[3 6; 5 7] * iso[10; 7] * J[2 8; 6 10] * conj(state.AR[j][1 8; 9])

    elseif k < l < i < j
        iso = isomorphism(flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] * iso[5; 3] * 
                            τ[5 6; 4 -2] * conj(state.AC[k][1 6; -1])
        τ1 = TensorMap(BraidingTensor(codomain(K, 2), domain(K, 2)))
        τ2 = TensorMap(BraidingTensor(codomain(K, 2), dual(flip(codomain(K, 1)))))
        if l > (k + 1)
            for a in (k + 1):(l - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, state.AR[a])
            end
        end
        @plansor Vₗ[-1 -2; -3] := Vₖ[1 2 3; 4] * state.AR[l][4 6; -3] * L[3 7; 6 5] * 
                            U[5] * τ2[2 8; 7 -2] * conj(state.AR[l][1 8; -1])
        if i > (l + 1)
            midsites = (l + 1):(i - 1)
            Vₗ = Vₗ * TransferMatrix(state.AR[midsites], fill(τ2, length(midsites)), state.AR[midsites])
        end
        @plansor Vᵢ[-1 -2 -3; -4] := Vₗ[1 2; 3] * state.AR[i][3 4; -4] * τ2[2 6; 4 -3] * 
                            I[5 7; 6 -2] * conj(U[5]) * conj(state.AR[i][1 7; -1])
        if j > (i + 1)
            τ3 = TensorMap(BraidingTensor(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(j - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vᵢ[1 2 3; 4] * state.AR[j][4 5; 9] * τ[3 6; 5 7] * iso[10; 7] * J[2 8; 6 10] * 
                            conj(state.AR[j][1 8; 9]) 
    else
        throw(ArgumentError("invalid input indices (i, j, k, l) for ($i, $j, $k, $l), only (i < j) && (k < l) is valid"))
    end
    return G
end