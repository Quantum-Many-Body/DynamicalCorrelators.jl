"""
    add_single_util_leg(tensor::AbstractTensorMap)

Add a trivial (one-dimensional) utility leg to a tensor map to balance the number of
codomain and domain legs. If `codomain < domain`, adds a leg to the codomain (left);
if `codomain > domain`, adds a leg to the domain (right).
This is needed when converting operators to MPO format.
"""
function add_single_util_leg(tensor::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    ou = oneunit(_firstspace(tensor))
    if length(codomain(tensor)) < length(domain(tensor))
        util = isomorphism(storagetype(tensor), ou * codomain(tensor), codomain(tensor))
        tensors = util * tensor
    elseif length(codomain(tensor)) > length(domain(tensor))
        util = isomorphism(storagetype(tensor), domain(tensor), domain(tensor) * ou)
        tensors = tensor * util
    else
        throw(ArgumentError("invalid operator"))
    end
    return tensors
end

"""
    cart2polar(point::AbstractArray) -> (r, θ, ϕ)

Convert Cartesian coordinates to polar/spherical coordinates.
For 2D points, returns `(r, θ, π/2)`. For 3D points, returns `(r, θ, ϕ)`.
"""
function cart2polar(point::AbstractArray)
    r = norm(point)
    θ = atan(point[2], point[1])
    ϕ = length(point) == 3 ? acos(point[3]/r) : π/2
    return r, θ, ϕ
end

"""
    phase_by_polar(theta, phi, phases)

Return a closure that maps a `Bond` to its phase factor based on its polar angle.
Used to assign direction-dependent hopping amplitudes (e.g., d-wave pairing).

# Arguments
- `theta`: array of arrays of θ values for each phase group.
- `phi`: array of arrays of ϕ values for each phase group.
- `phases`: the phase factor to assign for each group.
"""
function phase_by_polar(theta::AbstractVector, phi::AbstractVector, phases::AbstractVector)
    function _phase_by_polar(bond::Bond)
        _, θ, ϕ = cart2polar(rcoordinate(bond))
        for i in eachindex(phases)
            any(≈(θ), theta[i]) && any(≈(ϕ), phi[i]) && return phases[i]
        end
    end
    return _phase_by_polar
end

"""
    sort_by_distance(latt::CustomLattice, ij)

Given a flat array `ij` of site indices (first half = left sites, second half = right sites),
compute all pairwise distances and return the site pairs sorted by distance.

Returns `(is, js, ds)` where `is[k]`-`js[k]` is the k-th pair and `ds[k]` is its distance.
"""
function sort_by_distance(latt::CustomLattice, ij)
    # Split ij into left (reversed) and right halves
    is = ij[length(ij)÷2:-1:1]
    js = ij[length(ij)÷2+1:1:length(ij)]
    # Also form cross-pairs from adjacent elements
    ks = is[2:end]
    ls = js[1:end-1]
    r = []
    for i in eachindex(is)
        push!(r, [is[i], js[i], norm(latt.lattice[is[i]]-latt.lattice[js[i]])])
    end
    for i in eachindex(ks)
        push!(r, [ks[i], ls[i], norm(latt.lattice[ks[i]]-latt.lattice[ls[i]])])
    end
    sort!(r, by=x->x[3])
    is = zeros(Int64, length(r))
    js = zeros(Int64, length(r))
    ds = zeros(Float64, length(r))
    for i in eachindex(is)
        is[i] = Int(r[i][1])
        js[i] = Int(r[i][2])
        ds[i] = r[i][3]
    end
    return is, js, ds
end

"""
    transfer_left(v, A, Ab)

Left transfer operation for a 4-leg boundary tensor `v` through an MPS tensor `A`
and its conjugate `Ab`. Used in multi-site correlation function contractions
where extra operator legs need to be propagated through the chain via braiding tensors (τ).
"""
function transfer_left(v::AbstractTensorMap{T, S, 4, 1}, A::MPSTensor{S}, Ab::MPSTensor{S}) where {T, S}
    check_unambiguous_braiding(space(v, 4))
    @plansor v[-1 -2 -3 -4; -5] := v[1 2 3 4; 5] * A[5 6; -5] * τ[4 7; 6 -4] * τ[3 8; 7 -3] * τ[2 9; 8 -2] * conj(Ab[1 9; -1])
end

"""
    transfer_left(v, A, Ab)

Left transfer operation for a 3-leg boundary tensor `v`. See the 4-leg version above.
"""
function transfer_left(v::AbstractTensorMap{T, S, 3, 1}, A::MPSTensor{S}, Ab::MPSTensor{S}) where {T, S}
    check_unambiguous_braiding(space(v, 2))
    @plansor v[-1 -2 -3; -4] := v[1 2 3; 4] * A[4 5; -4] * τ[3 6; 5 -3] * τ[2 7; 6 -2] * conj(Ab[1 7; -1])
end


"""
    contract_MPO(mpo1::FiniteMPO, mpo2::FiniteMPO)

Vertically contract (fuse) two `FiniteMPO`s into a single `FiniteMPO`.
The boundary tensors are handled separately with explicit fusion, while
interior tensors use `fuse_mul_mpo`. The result is truncated with `notrunc()`.
"""
function contract_MPO(mpo1::FiniteMPO{<:MPOTensor}, mpo2::FiniteMPO{<:MPOTensor})
    T = promote_type(scalartype(mpo1[1]), scalartype(mpo2[end]))
    F1 = fuser(T, right_virtualspace(mpo2[1]), right_virtualspace(mpo1[1]))
    F2 = fuser(T, left_virtualspace(mpo2[end]), left_virtualspace(mpo1[end]))
    ops = map(fuse_mul_mpo, parent(mpo1)[2:end-1], parent(mpo2)[2:end-1])
    @plansor O₁[-1; -2 -3] :=  mpo2[1][1 2; -2 3] * mpo1[1][1 -1; 2 4] * conj(F1[-3; 3 4])
    @plansor O₂[-3 -1; -2] := F2[-3; 3 4] * mpo2[end][3 2; -2 1] * mpo1[end][4 -1; 2 1]
    O₁, O₂ = add_single_util_leg(O₁), add_single_util_leg(O₂)
    return changebonds!(FiniteMPO([O₁;ops;O₂]), SvdCut(; trscheme=notrunc()))
end
