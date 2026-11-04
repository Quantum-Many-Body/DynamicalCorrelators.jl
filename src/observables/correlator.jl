"""
    abstract type AbstractPlanChannel end

Internal supertype for the channels that can be registered in a [`CorrelationPlan`](@ref).
"""
abstract type AbstractPlanChannel end

"""
    OneSiteChannel(operator, indices)

One-site channel of a [`CorrelationPlan`](@ref): the local expectation of `operator`
at each lattice site, summed over the MPS indices of the site. The result is a
`Vector{ComplexF64}` over the lattice sites. See [`add_one_site!`](@ref).
"""
struct OneSiteChannel{O<:AbstractTensorMap} <: AbstractPlanChannel
    operator::O
    indices::Vector{Vector{Int}}
end

"""
    TwoSiteChannel(operators, indices, is, js)

Two-site channel of a [`CorrelationPlan`](@ref):
`⟨operators[1](i) operators[2](j)⟩` for `i in is` and `j in js`, summed over the MPS
indices of each lattice site. The result is a `Matrix{ComplexF64}` of size
`(length(is), length(js))`. See [`add_two_site!`](@ref).
"""
struct TwoSiteChannel{A<:AbstractTensorMap, B<:AbstractTensorMap} <: AbstractPlanChannel
    operators::Tuple{A, B}
    indices::Vector{Vector{Int}}
    is::Vector{Int}
    js::Vector{Int}
end

"""
    PairChannel(operators, amplitudes, indices, is, js)

Pair–pair channel of a [`CorrelationPlan`](@ref):
`result[a, b] = Σ_{p,q} conj(amplitude_p) * amplitude_q * ⟨operators[1](p) operators[2](q)⟩`,
where `p` and `q` run over the index tuples (bonds or on-site tuples) attached to
lattice sites `is[a]` and `js[b]`. See [`add_pairing!`](@ref).
"""
struct PairChannel{A<:AbstractTensorMap, B<:AbstractTensorMap} <: AbstractPlanChannel
    operators::Tuple{A, B}
    amplitudes::Vector{Vector{ComplexF64}}
    indices::Vector{Vector{Vector{Int}}}
    is::Vector{Int}
    js::Vector{Int}
end

"""
    CorrelationPlan(gs::AbstractFiniteMPS)

Lazy plan for static correlation functions of a finite-MPS ground state `gs`.

Channels are registered with [`add_one_site!`](@ref), [`add_two_site!`](@ref) and
[`add_pairing!`](@ref). Registration only records the operators and the geometry; no
tensor contraction happens at this point. [`correlator(plan)`](@ref) then computes the
raw expectation values required by the registered channels, caches them by operator
identity and MPS site indices (amplitudes are not part of the cache key, so channels
that differ only in their form factors share the same raw values), and returns the
channel results as a `NamedTuple` keyed by channel name.

```julia
plan = CorrelationPlan(gs)
add_one_site!(plan, :density; operator=n, lattice=bl, orbital=1)
add_two_site!(plan, :spin; operators=(sp, sm), lattice=bl, orbital=1)
add_pairing!(plan, :d_plus_id; operators=(sld, sl), lattice=bl,
    neighbors=Neighbors(1=>1), orbitals=(1, 1), amplitude=f)
results = correlator(plan)
```

Calling `correlator(plan)` again reuses the cached raw values; registering another
channel afterwards only triggers the raw work that is still missing. The number of raw
kernel evaluations can be inspected with [`raw_call_counts`](@ref).
"""
mutable struct CorrelationPlan{M<:AbstractFiniteMPS}
    gs::M
    names::Vector{Symbol}
    channels::Dict{Symbol, AbstractPlanChannel}
    operators::Vector{Any}
    opids::IdDict{Any, Int}
    cache1::Dict{Tuple{Int, Int}, ComplexF64}
    cache2::Dict{NTuple{4, Int}, ComplexF64}
    cache4::Dict{Tuple{Int, Int, NTuple{2, Int}, NTuple{2, Int}}, ComplexF64}
    counters::NamedTuple{(:onesite, :twosite, :pair), NTuple{3, Base.RefValue{Int}}}
end

function CorrelationPlan(gs::AbstractFiniteMPS)
    return CorrelationPlan(gs, Symbol[], Dict{Symbol, AbstractPlanChannel}(), Any[],
        IdDict{Any, Int}(), Dict{Tuple{Int, Int}, ComplexF64}(),
        Dict{NTuple{4, Int}, ComplexF64}(),
        Dict{Tuple{Int, Int, NTuple{2, Int}, NTuple{2, Int}}, ComplexF64}(),
        (onesite=Ref(0), twosite=Ref(0), pair=Ref(0)))
end

"""
    raw_call_counts(plan::CorrelationPlan)

Return the number of raw kernel evaluations performed so far, per channel kind
(`(onesite=…, twosite=…, pair=…)`). Useful for tests and diagnostics.
"""
raw_call_counts(plan::CorrelationPlan) = map(getindex, plan.counters)

function _opid!(plan::CorrelationPlan, op)
    return get!(plan.opids, op) do
        push!(plan.operators, op)
        length(plan.operators)
    end
end

function _register!(plan::CorrelationPlan, name::Symbol, channel::AbstractPlanChannel)
    haskey(plan.channels, name) && throw(ArgumentError("channel :$name is already registered"))
    plan.channels[name] = channel
    push!(plan.names, name)
    return plan
end

function _check_sites(is, js, N)
    all(i -> 1 <= i <= N, is) || throw(ArgumentError("is contains sites outside 1:$N"))
    all(j -> 1 <= j <= N, js) || throw(ArgumentError("js contains sites outside 1:$N"))
    return nothing
end

"""
    _site_indices(latt::CustomLattice; a::Union{Nothing, Integer})
"""
function _site_indices(latt::CustomLattice; a::Union{Nothing, Integer})
    indices = isnothing(a) ? latt.indices : [[latt.indices[i][a],] for i in 1:length(latt.lattice)]
    return indices
end

"""
    _pair_amplitude_indices(latt::CustomLattice, neighbors::Neighbors, a::Integer, b::Integer; amplitude::Union{Nothing, Function}=nothing, intralayer::Union{Nothing, Bool}=nothing)
"""
function _pair_amplitude_indices(latt::CustomLattice, neighbors::Neighbors, a::Integer, b::Integer; amplitude::Union{Nothing, Function}=nothing, intralayer::Union{Nothing, Bool}=nothing)
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
    add_one_site!(plan::CorrelationPlan, name::Symbol; operator, lattice, orbital=nothing)

Register a one-site channel: the local expectation of `operator` at each lattice site,
summed over the site's MPS indices. `orbital` selects one orbital; `nothing` sums over
all orbitals of each site. No tensor contraction happens until `correlator(plan)`.
The result is a `Vector{ComplexF64}` over the lattice sites.
"""
function add_one_site!(plan::CorrelationPlan, name::Symbol; operator::AbstractTensorMap,
        lattice::CustomLattice, orbital::Union{Nothing, Integer}=nothing)
    return _register!(plan, name, OneSiteChannel(operator, _site_indices(lattice; a=orbital)))
end

"""
    add_two_site!(plan::CorrelationPlan, name::Symbol; operators, lattice, orbital=nothing, is=1:L, js=1:L)

Register a two-site channel:
`result[a, b] = Σ ⟨operators[1](i) operators[2](j)⟩` with `i = is[a]`, `j = js[b]`,
summed over the MPS indices of each lattice site. `orbital` selects one orbital;
`nothing` sums over all orbitals. No tensor contraction happens until
`correlator(plan)`. The result is a `Matrix{ComplexF64}` of size
`(length(is), length(js))`.
"""
function add_two_site!(plan::CorrelationPlan, name::Symbol;
        operators::Tuple{<:AbstractTensorMap, <:AbstractTensorMap}, lattice::CustomLattice,
        orbital::Union{Nothing, Integer}=nothing,
        is::AbstractVector{<:Integer}=1:length(lattice.lattice),
        js::AbstractVector{<:Integer}=1:length(lattice.lattice))
    is, js = collect(is), collect(js)
    _check_sites(is, js, length(lattice.lattice))
    return _register!(plan, name,
        TwoSiteChannel(Tuple(operators), _site_indices(lattice; a=orbital), is, js))
end

"""
    add_pairing!(plan::CorrelationPlan, name::Symbol; operators, lattice, neighbors, orbitals=(1, 1), amplitude=nothing, intralayer=nothing, is=1:L, js=1:L)

Register a pair–pair channel:
`result[a, b] = Σ_{p,q} conj(amplitude_p) * amplitude_q * ⟨operators[1](p) operators[2](q)⟩`,
where `p` and `q` run over the index tuples of `neighbors` bonds attached to lattice
sites `is[a]` and `js[b]`, and `orbitals` selects the orbital of each leg of a bond.
`amplitude` is a function of the bond (the form factor); `intralayer` filters bonds by
layer. On-site pairing uses `neighbors=Neighbors(0=>0)` with on-site pair operators.
No tensor contraction happens until `correlator(plan)`. The result is a
`Matrix{ComplexF64}` of size `(length(is), length(js))`.
"""
function add_pairing!(plan::CorrelationPlan, name::Symbol;
        operators::Tuple{<:AbstractTensorMap, <:AbstractTensorMap}, lattice::CustomLattice,
        neighbors::Neighbors, orbitals::Tuple{<:Integer, <:Integer}=(1, 1),
        amplitude::Union{Nothing, Function}=nothing, intralayer::Union{Nothing, Bool}=nothing,
        is::AbstractVector{<:Integer}=1:length(lattice.lattice),
        js::AbstractVector{<:Integer}=1:length(lattice.lattice))
    is, js = collect(is), collect(js)
    _check_sites(is, js, length(lattice.lattice))
    amplitudes, indices = _pair_amplitude_indices(lattice, neighbors, orbitals[1], orbitals[2];
        amplitude=amplitude, intralayer=intralayer)
    return _register!(plan, name, PairChannel(Tuple(operators),
        [ComplexF64.(a) for a in amplitudes], indices, is, js))
end

"""
    correlator(plan::CorrelationPlan; parallel::Integer=Threads.nthreads())

Compute all registered channels and return a `NamedTuple` keyed by channel name. Raw
expectation values are shared between channels through the plan caches: each raw value
is computed only once, and only the values that are still missing are evaluated, using
`parallel` threads on shared memory.
"""
function correlator(plan::CorrelationPlan; parallel::Integer=Threads.nthreads())
    req1 = Set{Tuple{Int, Int}}()
    req2 = Set{NTuple{4, Int}}()
    req4 = Set{Tuple{Int, Int, NTuple{2, Int}, NTuple{2, Int}}}()
    for name in plan.names
        _collect_requirements!(req1, req2, req4, plan, plan.channels[name])
    end
    _fill_cache1!(plan, req1, parallel)
    _fill_cache2!(plan, req2, parallel)
    _fill_cache4!(plan, req4, parallel)
    return NamedTuple{Tuple(plan.names)}(Tuple(
        _channel_result(plan, plan.channels[name]) for name in plan.names))
end

function _collect_requirements!(req1, req2, req4, plan::CorrelationPlan, ch::OneSiteChannel)
    opid = _opid!(plan, ch.operator)
    for idxs in ch.indices, i in idxs
        push!(req1, (opid, i))
    end
    return nothing
end

function _collect_requirements!(req1, req2, req4, plan::CorrelationPlan, ch::TwoSiteChannel)
    id1, id2 = _opid!(plan, ch.operators[1]), _opid!(plan, ch.operators[2])
    for ii in ch.is, jj in ch.js, i in ch.indices[ii], j in ch.indices[jj]
        push!(req2, (id1, id2, i, j))
    end
    return nothing
end

function _collect_requirements!(req1, req2, req4, plan::CorrelationPlan, ch::PairChannel)
    id1, id2 = _opid!(plan, ch.operators[1]), _opid!(plan, ch.operators[2])
    for ii in ch.is, jj in ch.js, p in ch.indices[ii], q in ch.indices[jj]
        if length(p) == 1
            push!(req2, (id1, id2, p[1], q[1]))
        else
            push!(req4, (id1, id2, Tuple(p), Tuple(q)))
        end
    end
    return nothing
end

function _parallel_eval(f, n::Integer, parallel::Integer)
    out = Vector{ComplexF64}(undef, n)
    if parallel == 1
        for t in 1:n
            out[t] = f(t)
        end
    else
        pos = Threads.Atomic{Int}(1)
        Threads.@sync for _ in 1:parallel
            Threads.@spawn while true
                t = Threads.atomic_add!(pos, 1)
                t > n && break
                out[t] = f(t)
            end
        end
    end
    return out
end

function _fill_cache1!(plan::CorrelationPlan, req, parallel::Integer)
    missing = sort!(collect(setdiff(req, keys(plan.cache1))))
    isempty(missing) && return nothing
    ψ, ops = plan.gs, plan.operators
    vals = _parallel_eval(length(missing), parallel) do t
        key = missing[t]
        ComplexF64(correlator(ψ, ops[key[1]], key[2]))
    end
    for t in eachindex(missing)
        plan.cache1[missing[t]] = vals[t]
    end
    plan.counters.onesite[] += length(missing)
    return nothing
end

function _fill_cache2!(plan::CorrelationPlan, req, parallel::Integer)
    missing = sort!(collect(setdiff(req, keys(plan.cache2))))
    isempty(missing) && return nothing
    ψ, ops = plan.gs, plan.operators
    vals = _parallel_eval(length(missing), parallel) do t
        key = missing[t]
        ComplexF64(correlator(ψ, ops[key[1]], ops[key[2]], (key[3],), (key[4],)))
    end
    for t in eachindex(missing)
        plan.cache2[missing[t]] = vals[t]
    end
    plan.counters.twosite[] += length(missing)
    return nothing
end

function _fill_cache4!(plan::CorrelationPlan, req, parallel::Integer)
    missing = sort!(collect(setdiff(req, keys(plan.cache4))))
    isempty(missing) && return nothing
    ψ, ops = plan.gs, plan.operators
    vals = _parallel_eval(length(missing), parallel) do t
        key = missing[t]
        ComplexF64(correlator(ψ, ops[key[1]], ops[key[2]], key[3], key[4]))
    end
    for t in eachindex(missing)
        plan.cache4[missing[t]] = vals[t]
    end
    plan.counters.pair[] += length(missing)
    return nothing
end

function _channel_result(plan::CorrelationPlan, ch::OneSiteChannel)
    opid = plan.opids[ch.operator]
    return [sum((plan.cache1[(opid, i)] for i in idxs); init=0.0im) for idxs in ch.indices]
end

function _channel_result(plan::CorrelationPlan, ch::TwoSiteChannel)
    id1, id2 = plan.opids[ch.operators[1]], plan.opids[ch.operators[2]]
    return [sum((plan.cache2[(id1, id2, i, j)] for i in ch.indices[ii], j in ch.indices[jj]);
                init=0.0im)
            for ii in ch.is, jj in ch.js]
end

function _channel_result(plan::CorrelationPlan, ch::PairChannel)
    id1, id2 = plan.opids[ch.operators[1]], plan.opids[ch.operators[2]]
    return [sum((_pair_raw(plan, id1, id2, p, q) * conj(a) * b
                for (p, a) in zip(ch.indices[ii], ch.amplitudes[ii]),
                    (q, b) in zip(ch.indices[jj], ch.amplitudes[jj])); init=0.0im)
            for ii in ch.is, jj in ch.js]
end

function _pair_raw(plan::CorrelationPlan, id1, id2, p, q)
    if length(p) == 1
        return plan.cache2[(id1, id2, p[1], q[1])]
    else
        return plan.cache4[(id1, id2, Tuple(p), Tuple(q))]
    end
end

function correlator(gs::AbstractFiniteMPS, O₁::AbstractTensorMap, O₂::AbstractTensorMap; is=1:length(gs), js=1:length(gs), parallel=Threads.nthreads())
    T = promote_type(scalartype(gs), scalartype(O₁), scalartype(O₂))
    Fr = zeros(T, length(is), length(js))
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

Compute the two-operator correlation. With `O₁ = e_min(...; side=:L)` and
`O₂ = e_plus(...; side=:R)`, the tensor convention returns `-⟨c_i c†_j⟩` for all
`i` and `j`. Multiply the result by `-1` to obtain the directly ordered `⟨c_i c†_j⟩`.
"""
function correlator(state::AbstractFiniteMPS, O₁::AbstractTensorMap, O₂::AbstractTensorMap, i::NTuple{1, Integer}, j::NTuple{1, Integer})
    i, j = i[1], j[1]
    T = promote_type(scalartype(state), scalartype(O₁), scalartype(O₂))
    if (length(domain(O₁)) == 2)&&(length(codomain(O₂)) == 2)
        if i == j 
            O = contract_onesite(O₁, O₂)
            G = @plansor state.AC[i][1 2; 3] * O[4; 2] * conj(state.AC[i][1 4; 3])
        elseif  i < j
            @plansor Vₗ[-1 -2; -3] := state.AC[i][3 4; -3] * O₁[2; 4 -2] * conj(state.AC[i][3 2; -1])
            ctr = i + 1
            if j > ctr
                Z = TensorMap(BraidingTensor{T}(domain(O₁,1), domain(O₁,2)))
                midsites = ctr:(j - 1)
                Vₗ = Vₗ * TransferMatrix(state.AR[midsites], fill(Z, length(midsites)), state.AR[midsites])
            end
            G = @plansor Vₗ[2 3; 5] * state.AR[j][5 6; 7] * O₂[3 4; 6] * conj(state.AR[j][2 4; 7])
        else
            iso₂ = isomorphism(T, flip(codomain(O₂, 1)), codomain(O₂, 1))
            @plansor Vₗ[-1 -2; -3] := state.AC[j][3 4; -3] * O₂[2 1; 4] * iso₂[6; 2] * τ[6 5; 1 -2] * conj(state.AC[j][3 5; -1])
            ctr = j + 1
            if i > ctr
                Z = TensorMap(BraidingTensor{T}(domain(O₁,1), dual(flip(codomain(O₂, 1)))))
                midsites = ctr:(i - 1)
                Vₗ = Vₗ * TransferMatrix(state.AR[midsites], fill(Z, length(midsites)), state.AR[midsites])
            end
            iso₁ = isomorphism(T, codomain(O₂, 1), flip(codomain(O₂, 1)))
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
        # TensorKit >= 0.17 forbids mixing storagetypes in one contraction, so the
        # braiding/isomorphism helpers below must carry the ambient scalar type
        T = promote_type(scalartype(state), scalartype(O₁), scalartype(O₂))
        i, j = ij
        k, l = kl
    if i < j < k < l
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if j > (i + 1)
            Zᵢⱼ = TensorMap(BraidingTensor{T}(domain(I,1), domain(I, 2)))
            midsites = (i + 1):(j - 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[midsites], fill(Zᵢⱼ, length(midsites)), state.AR[midsites])
        end
        @plansor Vⱼ[-1 -2; -3] := Vᵢ[1 2; 3] * state.AR[j][3 4; -3] * J[2 5; 4 -2] *
                            conj(state.AR[j][1 5; -1])
        if k > (j + 1)
            Zⱼₖ = TensorMap(BraidingTensor{T}(domain(J, 1), domain(J, 2)))
            midsites = (j + 1):(k - 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[midsites], fill(Zⱼₖ, length(midsites)), state.AR[midsites])
        end
        @plansor Vₖ[-1 -2; -3] := Vⱼ[1 2; 3] * state.AR[k][3 4; -3] * K[2 5; 4 -2] *
                            conj(state.AR[k][1 5; -1])
        if l > (k + 1)
            Zₖₗ = TensorMap(BraidingTensor{T}(domain(K, 1), domain(K, 2)))
            midsites = (k + 1):(l - 1)
            Vₖ = Vₖ * TransferMatrix(state.AR[midsites], fill(Zₖₗ, length(midsites)), state.AR[midsites])
        end
        G = @plansor Vₖ[2 3; 5] * state.AR[l][5 6; 7] * L[3 4; 6 1] * U[1] *
                        conj(state.AR[l][2 4; 7])

    elseif i < j == k < l
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if j > (i + 1)
            Zᵢⱼ = TensorMap(BraidingTensor{T}(domain(I,1), domain(I, 2)))
            midsites = (i + 1):(j - 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[midsites], fill(Zᵢⱼ, length(midsites)), state.AR[midsites])
        end
        @plansor Vⱼ[-1 -2; -3] := Vᵢ[1 2; 3] * state.AR[j][3 4; -3] * K[5 6; 4 -2] *
                            τ[5 7; 6 8] * J[2 9; 7 8] * conj(state.AR[j][1 9; -1])
        if l > (j + 1)
            Zₖₗ = TensorMap(BraidingTensor{T}(domain(K, 1), domain(K, 2)))
            midsites = (j + 1):(l - 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[midsites], fill(Zₖₗ, length(midsites)), state.AR[midsites])
        end
        G = @plansor Vⱼ[2 3; 5] * state.AR[l][5 6; 7] * L[3 4; 6 1] * U[1] *
                            conj(state.AR[l][2 4; 7])

    elseif i < k < j < l
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if k > (i + 1)
            Zᵢₖ = TensorMap(BraidingTensor{T}(domain(I,1), domain(I, 2)))
            midsites = (i + 1):(k - 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[midsites], fill(Zᵢₖ, length(midsites)), state.AR[midsites])
        end                     
        iso = isomorphism(T, flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vₖ[-1 -2 -3 -4; -5] := Vᵢ[1 2; 3] * state.AR[k][3 4; -5] * 
                            K[5 6; 4 -4] * iso[9; 5] * τ[9 7; 6 -3] * τ[2 8; 7 -2] * conj(state.AR[k][1 8; -1])
        if j > (k + 1)
            τ1 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(K, 2)))
            τ2 = TensorMap(BraidingTensor{T}(codomain(K, 2), dual(flip(codomain(K, 1)))))
            τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
            for a in (k + 1):(j - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(T, codomain(K, 1), flip(codomain(K, 1)))
        @plansor Vⱼ[-1, -2; -3] := Vₖ[1 2 3 4; 5] * state.AR[j][5 6; -3] * τ[4 7; 6 -2] * 
                            τ[3 8; 7 9] * J[2 10; 8 11] * iso[11; 9] * conj(state.AR[j][1 10; -1])
        if l > (j + 1)
            Zⱼₗ = TensorMap(BraidingTensor{T}(domain(K, 1), domain(K, 2)))
            midsites = (j + 1):(l - 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[midsites], fill(Zⱼₗ, length(midsites)), state.AR[midsites])
        end
        G = @plansor Vⱼ[2 3; 5] * state.AR[l][5 6; 7] * L[3 4; 6 1] * U[1] *
                            conj(state.AR[l][2 4; 7])
    
    elseif i < k < j == l
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if k > (i + 1)
            Zᵢₖ = TensorMap(BraidingTensor{T}(domain(I,1), domain(I, 2)))
            midsites = (i + 1):(k - 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[midsites], fill(Zᵢₖ, length(midsites)), state.AR[midsites])
        end                     
        iso = isomorphism(T, flip(codomain(K, 1)), codomain(K, 1))
        τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
        @plansor Vₖ[-1 -2 -3 -4; -5] := Vᵢ[1 2; 3] * state.AR[k][3 4; -5] * 
                            K[5 6; 4 -4] * iso[9; 5] * τ[9 7; 6 -3] * τ3[2 8; 7 -2] * conj(state.AR[k][1 8; -1])
        if j > (k + 1)
            τ1 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(K, 2)))
            τ2 = TensorMap(BraidingTensor{T}(codomain(K, 2), dual(flip(codomain(K, 1)))))
            τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
            for a in (k + 1):(j - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(T, codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vₖ[1 2 3 4; 5] * state.AR[j][5 6; 13] * L[4 7; 6 12] * U[12] *
                            τ[3 8; 7 9] * iso[11; 9] * J[2 10; 8 11] * conj(state.AR[j][1 10; 13])

    elseif i == k < j < l
        iso = isomorphism(T, flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vᵢ[-1 -2 -3 -4; -5] := state.AC[i][3 7; -5] * K[5 6; 7 -4] * iso[8; 5] * τ[8 4; 6 -3] *
                            conj(U[1]) * I[1 2; 4 -2] * conj(state.AC[i][3 2; -1])
        if j > (i + 1)
            τ1 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(K, 2)))
            τ2 = TensorMap(BraidingTensor{T}(codomain(K, 2), dual(flip(codomain(K, 1)))))
            τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(j - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(T, codomain(K, 1), flip(codomain(K, 1)))
        τ1 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(K, 2)))
        @plansor Vⱼ[-1, -2; -3] := Vᵢ[1 2 3 4; 5] * state.AR[j][5 6; -3] * τ1[4 7; 6 -2] * 
                            τ[3 8; 7 9] * J[2 10; 8 11] * iso[11; 9] * conj(state.AR[j][1 10; -1])
        if l > (j + 1)
            Zⱼₗ = TensorMap(BraidingTensor{T}(domain(K, 1), domain(K, 2)))
            midsites = (j + 1):(l - 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[midsites], fill(Zⱼₗ, length(midsites)), state.AR[midsites])
        end
        G = @plansor Vⱼ[2 3; 5] * state.AR[l][5 6; 7] * L[3 4; 6 1] * U[1] *
                            conj(state.AR[l][2 4; 7])
    
    elseif i == k < j == l
        iso = isomorphism(T, flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vᵢ[-1 -2 -3 -4; -5] := state.AC[i][3 7; -5] * K[5 6; 7 -4] * iso[8; 5] * τ[8 4; 6 -3] *
                            conj(U[1]) * I[1 2; 4 -2] * conj(state.AC[i][3 2; -1])
        if j > (i + 1)
            τ1 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(K, 2)))
            τ2 = TensorMap(BraidingTensor{T}(codomain(K, 2), dual(flip(codomain(K, 1)))))
            τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(j - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(T, codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vᵢ[1 2 3 4; 5] * state.AR[j][5 6; 13] * L[4 7; 6 12] * U[12] *
                            τ[3 8; 7 9] * iso[11; 9] * J[2 10; 8 11] * conj(state.AR[j][1 10; 13])
    
    elseif i < k < l < j
        @plansor Vᵢ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * I[1 2; 4 -2] *
                            conj(state.AC[i][3 2; -1])
        if k > (i + 1)
            Zᵢₖ = TensorMap(BraidingTensor{T}(domain(I,1), domain(I, 2)))
            midsites = (i + 1):(k - 1)
            Vᵢ = Vᵢ * TransferMatrix(state.AR[midsites], fill(Zᵢₖ, length(midsites)), state.AR[midsites])
        end                     
        iso = isomorphism(T, flip(codomain(K, 1)), codomain(K, 1))
        τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
        @plansor Vₖ[-1 -2 -3 -4; -5] := Vᵢ[1 2; 3] * state.AR[k][3 4; -5] * 
                            K[5 6; 4 -4] * iso[9; 5] * τ[9 7; 6 -3] * τ3[2 8; 7 -2] * conj(state.AR[k][1 8; -1])
        if l > (k + 1)
            τ1 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(K, 2)))
            τ2 = TensorMap(BraidingTensor{T}(codomain(K, 2), dual(flip(codomain(K, 1)))))
            τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
            for a in (k + 1):(l - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        τ2 = TensorMap(BraidingTensor{T}(codomain(K, 2), dual(flip(codomain(K, 1)))))
        @plansor Vₗ[-1 -2 -3; -4] := Vₖ[1 2 3 4; 5] * state.AR[l][5 6; -4] * L[4 7; 6 10] * U[10] *
                            τ2[3 8; 7 -3] * τ3[2 9; 8 -2] * conj(state.AR[l][1 9; -1])
        if j > (l + 1)
            for a in (l + 1):(j - 1)
                Vₗ = transfer_left(Vₗ, state.AR[a], τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(T, codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vₗ[1 2 3; 4] * state.AR[j][4 5; 9] *
                            τ[3 6; 5 7] * iso[10; 7] * J[2 8; 6 10] * conj(state.AR[j][1 8; 9])

    elseif i == k < l < j
        iso = isomorphism(T, flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vᵢ[-1 -2 -3 -4; -5] := state.AC[i][3 7; -5] * K[5 6; 7 -4] * iso[8; 5] * τ[8 4; 6 -3] *
                            conj(U[1]) * I[1 2; 4 -2] * conj(state.AC[i][3 2; -1])
        if l > (i + 1)
            τ1 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(K, 2)))
            τ2 = TensorMap(BraidingTensor{T}(codomain(K, 2), dual(flip(codomain(K, 1)))))
            τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(l - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        τ2 = TensorMap(BraidingTensor{T}(codomain(K, 2), dual(flip(codomain(K, 1)))))
        τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
        @plansor Vₗ[-1 -2 -3; -4] := Vᵢ[1 2 3 4; 5] * state.AR[l][5 6; -4] * L[4 7; 6 10] * U[10] *
                            τ2[3 8; 7 -3] * τ3[2 9; 8 -2] * conj(state.AR[l][1 9; -1])
        if j > (l + 1)
            for a in (l + 1):(j - 1)
                Vₗ = transfer_left(Vₗ, state.AR[a], τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(T, codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vₗ[1 2 3; 4] * state.AR[j][4 5; 9] *
                            τ[3 6; 5 7] * iso[10; 7] * J[2 8; 6 10] * conj(state.AR[j][1 8; 9])

    elseif k < i < j < l
        iso = isomorphism(T, flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] * iso[5; 3] * 
                            τ[5 6; 4 -2] * conj(state.AC[k][1 6; -1])
        τ1 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(K, 2)))
        τ2 = TensorMap(BraidingTensor{T}(codomain(K, 2), dual(flip(codomain(K, 1)))))
        if i > (k + 1)
            for a in (k + 1):(i - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, state.AR[a])
            end
        end
        @plansor Vᵢ[-1 -2 -3 -4; -5] := Vₖ[1 2 3; 4] * state.AR[i][4 5; -5] * τ1[3 6; 5 -4] * 
                            τ2[2 8; 6 -3] * conj(U[7]) * I[7 9; 8 -2] * conj(state.AR[i][1 9; -1])
        if j > (i + 1)
            τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(j - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(T, codomain(K, 1), flip(codomain(K, 1)))
        @plansor Vⱼ[-1 -2; -3] := Vᵢ[1 2 3 4; 5] * state.AR[j][5 6; -3] * τ1[4 7; 6 -2] * τ2[3 8; 7 9] *
                            J[2 10; 8 11] * iso[11; 9] * conj(state.AR[j][1 10; -1])
        if l > (j + 1)
            Zⱼₗ = TensorMap(BraidingTensor{T}(domain(K, 1), domain(K, 2)))
            midsites = (j + 1):(l - 1)
            Vⱼ = Vⱼ * TransferMatrix(state.AR[midsites], fill(Zⱼₗ, length(midsites)), state.AR[midsites])
        end
        G = @plansor Vⱼ[1 2; 3] * state.AR[l][3 4; 6] * L[2 5; 4 7] * U[7] * conj(state.AR[l][1 5; 6])

    elseif k < i < j == l
        iso = isomorphism(T, flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] * iso[5; 3] * 
                            τ[5 6; 4 -2] * conj(state.AC[k][1 6; -1])
        τ1 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(K, 2)))
        τ2 = TensorMap(BraidingTensor{T}(codomain(K, 2), dual(flip(codomain(K, 1)))))
        if i > (k + 1)
            for a in (k + 1):(i - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, state.AR[a])
            end
        end
        @plansor Vᵢ[-1 -2 -3 -4; -5] := Vₖ[1 2 3; 4] * state.AR[i][4 5; -5] * τ1[3 6; 5 -4] * 
                            τ2[2 8; 6 -3] * conj(U[7]) * I[7 9; 8 -2] * conj(state.AR[i][1 9; -1])
        if j > (i + 1)
            τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(j - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(T, codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vᵢ[1 2 3 4; 5] * state.AR[j][5 6; 12] * L[4 8; 6 7] * U[7] * τ[3 9; 8 10] * iso[13; 10] *
                            J[2 11; 9 13] * conj(state.AR[j][1 11; 12])
        
    elseif k < i < l < j
        iso = isomorphism(T, flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] * iso[5; 3] * 
                            τ[5 6; 4 -2] * conj(state.AC[k][1 6; -1])
        τ1 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(K, 2)))
        τ2 = TensorMap(BraidingTensor{T}(codomain(K, 2), dual(flip(codomain(K, 1)))))
        if i > (k + 1)
            for a in (k + 1):(i - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, state.AR[a])
            end
        end
        @plansor Vᵢ[-1 -2 -3 -4; -5] := Vₖ[1 2 3; 4] * state.AR[i][4 5; -5] * τ1[3 6; 5 -4] * 
                            τ2[2 8; 6 -3] * conj(U[7]) * I[7 9; 8 -2] * conj(state.AR[i][1 9; -1])
        if l > (i + 1)
            τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(l - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ1, τ2, τ3, state.AR[a])
            end
        end
        τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
        @plansor Vₗ[-1 -2 -3; -4] := Vᵢ[1 2 3 4; 5] * state.AR[l][5 6; -4] * L[4 7; 6 10] * U[10] *
                            τ2[3 8; 7 -3] * τ3[2 9; 8 -2] * conj(state.AR[l][1 9; -1])
        if j > (l + 1)
            for a in (l + 1):(j - 1)
                Vₗ = transfer_left(Vₗ, state.AR[a], τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(T, codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vₗ[1 2 3; 4] * state.AR[j][4 5; 9] *
                            τ[3 6; 5 7] * iso[10; 7] * J[2 8; 6 10] * conj(state.AR[j][1 8; 9])

    elseif k < i == l < j
        iso = isomorphism(T, flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] * iso[5; 3] * 
                            τ[5 6; 4 -2] * conj(state.AC[k][1 6; -1])
        τ1 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(K, 2)))
        τ2 = TensorMap(BraidingTensor{T}(codomain(K, 2), dual(flip(codomain(K, 1)))))
        if i > (k + 1)
            for a in (k + 1):(i - 1)
                Vₖ = transfer_left(Vₖ, state.AR[a], τ1, τ2, state.AR[a])
            end
        end
        @plansor Vᵢ[-1 -2 -3; -4] := Vₖ[1 2 3; 4] * state.AR[i][4 5; -4] * L[3 7; 5 6] * U[6] *
                            τ2[2 9; 7 -3] * I[8 10; 9 -2] * conj(U[8]) * conj(state.AR[i][1 10; -1])
        if j > (i + 1)
            τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(j - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(T, codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vᵢ[1 2 3; 4] * state.AR[j][4 5; 9] *
                            τ[3 6; 5 7] * iso[10; 7] * J[2 8; 6 10] * conj(state.AR[j][1 8; 9])

    elseif k < l < i < j
        iso = isomorphism(T, flip(codomain(K, 1)), codomain(K, 1))
        @plansor Vₖ[-1 -2 -3; -4] := state.AC[k][1 2; -4] * K[3 4; 2 -3] * iso[5; 3] * 
                            τ[5 6; 4 -2] * conj(state.AC[k][1 6; -1])
        τ1 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(K, 2)))
        τ2 = TensorMap(BraidingTensor{T}(codomain(K, 2), dual(flip(codomain(K, 1)))))
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
            τ3 = TensorMap(BraidingTensor{T}(codomain(K, 2), domain(I, 2)))
            for a in (i + 1):(j - 1)
                Vᵢ = transfer_left(Vᵢ, state.AR[a], τ2, τ3, state.AR[a])
            end
        end
        iso = isomorphism(T, codomain(K, 1), flip(codomain(K, 1)))
        G = @plansor Vᵢ[1 2 3; 4] * state.AR[j][4 5; 9] * τ[3 6; 5 7] * iso[10; 7] * J[2 8; 6 10] * 
                            conj(state.AR[j][1 8; 9]) 
    else
        throw(ArgumentError("invalid input indices (i, j, k, l) for ($i, $j, $k, $l), only (i < j) && (k < l) is valid"))
    end
    return G
end