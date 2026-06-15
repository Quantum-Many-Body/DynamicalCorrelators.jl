"""
    randFiniteMPS(elt, U1Irrep, U1Irrep, N; filling=(1,1), md=10, mz=nothing)

Create a random finite MPS with fℤ₂ × U(1) × U(1) symmetry (charge and spin conservation).

# Arguments
- `elt`: element type (e.g., `Float64`, `ComplexF64`).
- `N`: number of sites.
- `filling`: tuple `(P, Q)` defining the filling fraction `P/Q`. Requires `N` to be a multiple of `2Q`.
- `md`: bond dimension for each quantum number sector (default: 10).
- `mz`: maximum charge quantum number range. If `nothing`, determined from filling.
"""
function randFiniteMPS(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, N::Integer; filling=(1,1), md=10, mz=nothing)
    P,Q = filling
    # Determine the unit cell size from filling: odd P → 2Q sites per cell, even P → Q sites
    isodd(P)&&(mod(N, 2Q)==0) ? (k = 0:2Q) : iseven(P)&&(mod(N, 2Q)==0) ? (k = 0:Q) : throw(ArgumentError("invalid length for the filling"))
    isnothing(mz) ? (ℤ = -max(P,Q):max(P,Q)) : (ℤ = -mz:mz)
    I = FermionParity ⊠ U1Irrep ⊠ U1Irrep
    Vs = [_vspaces(U1Irrep, U1Irrep, P, Q, k[i], ℤ, I, md) for i in 2:length(k)]
    # Physical space: 4-dimensional (empty, double-occupied, spin-up, spin-down)
    Ps = Vect[I]((0,0,-P) => 1, (0,0,2*Q-P) => 1, (1,1,Q-P) => 1, (1,-1,Q-P) => 1)
    pspaces = repeat([Ps,], N)
    M = div(N, 2Q)
    M == 1 ? maxvspaces = Vs[1:end-1] : maxvspaces = repeat(Vs, M)[1:end-1]
    randmps = FiniteMPS(rand, elt, pspaces, maxvspaces)
    return randmps
end

"""
    randInfiniteMPS(elt, U1Irrep, U1Irrep, N; filling=(1,1), md=10, mz=nothing)

Create a random infinite MPS with fℤ₂ × U(1) × U(1) symmetry.
Arguments are the same as [`randFiniteMPS`](@ref) for the U(1)×U(1) case.
"""
function randInfiniteMPS(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, N::Integer; filling=(1,1), md=10, mz=nothing)
    P,Q = filling
    isodd(P)&&(mod(N, 2Q)==0) ? (k = 0:2Q) : iseven(P)&&(mod(N, 2Q)==0) ? (k = 0:Q) : throw(ArgumentError("invalid length for the filling"))
    isnothing(mz) ? (ℤ = -max(P,Q):max(P,Q)) : (ℤ = -mz:mz)
    I = FermionParity ⊠ U1Irrep ⊠ U1Irrep
    Vs = [_vspaces(U1Irrep, U1Irrep, P, Q, k[i], ℤ, I, md) for i in 2:length(k)]
    Ps = Vect[I]((0,0,-P) => 1, (0,0,2*Q-P) => 1, (1,1,Q-P) => 1, (1,-1,Q-P) => 1)
    pspaces = repeat([Ps,], N)
    M = div(N, 2Q)
    M == 1 ? maxvspaces = Vs : maxvspaces = repeat(Vs, M)
    return InfiniteMPS(rand, elt, pspaces, Vs)
end

"""
    _vspaces(U1Irrep, U1Irrep, P, Q, k, Z, I, md)

Construct the virtual space for bond `k` in a U(1)×U(1) symmetric MPS.
Enumerates all allowed quantum number sectors `(fermion_parity, Sz, N)` with
bond dimension `md` per sector.
"""
function _vspaces(::Type{U1Irrep}, ::Type{U1Irrep}, P, Q, k, Z, I, md)
    vs = []
    for z₁ in Z
        for z₂ in Z
            push!(vs, (0, 2*z₂, 2*z₁*Q-k*P))
            push!(vs, (1, 2*z₂+1, (2*z₁+1)*Q-k*P))
        end
    end
    vsp = Vect[I]([v => md for v in vs]...)
    return vsp
end

"""
    randFiniteMPS(elt, SU2Irrep, U1Irrep, N; filling=(1,1), md=10, mz=nothing)

Create a random finite MPS with fℤ₂ × SU(2) × U(1) symmetry (spin-rotation and charge conservation).

# Arguments
- `elt`: element type.
- `N`: number of sites.
- `filling`: tuple `(P, Q)` defining filling `P/Q`.
- `md`: bond dimension per sector (default: 10).
- `mz`: maximum quantum number range. If `nothing`, determined from filling.
"""
function randFiniteMPS(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, N::Integer; filling=(1,1), md=10, mz=nothing)
    P,Q = filling
    isodd(P)&&(mod(N, 2Q)==0) ? (k = 0:2Q) : iseven(P)&&(mod(N, 2Q)==0) ? (k = 0:Q) : throw(ArgumentError("invalid length for the filling"))
    if mz == nothing
        ℤ = -max(P,Q):max(P,Q)
        ℕ = 0:max(P,Q)
    else
        ℤ = -mz:mz
        ℕ = 0:mz
    end
    I = FermionParity ⊠ SU2Irrep ⊠ U1Irrep
    Vs = [_vspaces(SU2Irrep, U1Irrep, P, Q, k[i], ℤ, ℕ, I, md) for i in 2:length(k)]
    Ps = Vect[I]((0,0,-P) => 1, (0,0,2*Q-P) => 1, (1,1//2,Q-P) => 1)
    pspaces = repeat([Ps,], N)
    M = div(N, length(Vs))
    M == 1 ? maxvspaces = Vs[1:end-1] : maxvspaces = repeat(Vs, M)[1:end-1]
    randmps = FiniteMPS(rand, elt, pspaces, maxvspaces)
    return randmps
end

"""
    randInfiniteMPS(elt, SU2Irrep, U1Irrep, N; filling=(1,1), md=10, mz=nothing)

Create a random infinite MPS with fℤ₂ × SU(2) × U(1) symmetry.
Arguments are the same as [`randFiniteMPS`](@ref) for the SU(2)×U(1) case.
"""
function randInfiniteMPS(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep}, N::Integer; filling=(1,1), md=10, mz=nothing)
    P,Q = filling
    isodd(P)&&(mod(N, 2Q)==0) ? (k = 0:2Q) : iseven(P)&&(mod(N, 2Q)==0) ? (k = 0:Q) : throw(ArgumentError("invalid length for the filling"))
    if mz == nothing
        ℤ = -max(P,Q):max(P,Q)
        ℕ = 0:max(P,Q)
    else
        ℤ = -mz:mz
        ℕ = 0:mz
    end
    I = FermionParity ⊠ SU2Irrep ⊠ U1Irrep
    Vs = [_vspaces(SU2Irrep, U1Irrep, P, Q, k[i], ℤ, ℕ, I, md) for i in 2:length(k)]
    Ps = Vect[I]((0,0,-P) => 1, (0,0,2*Q-P) => 1, (1,1//2,Q-P) => 1)
    pspaces = repeat([Ps,], N)
    M = div(N, length(Vs))
    M == 1 ? maxvspaces = Vs : maxvspaces = repeat(Vs, M)
    return InfiniteMPS(rand, elt, pspaces, Vs)
end

"""
    _vspaces(SU2Irrep, U1Irrep, P, Q, k, Z, N, I, md)

Construct the virtual space for bond `k` in an SU(2)×U(1) symmetric MPS.
Enumerates all allowed quantum number sectors `(fermion_parity, spin, charge)` with
bond dimension `md` per sector.
"""
function _vspaces(::Type{SU2Irrep}, ::Type{U1Irrep}, P, Q, k, Z, N, I, md)
    vs = []
    for z in Z
        for n in N
            push!(vs, (0, n, 2*z*Q-k*P))
            push!(vs, (1, n+1//2, (2*z+1)*Q-k*P))
        end
    end
    vsp = Vect[I]([v => md for v in vs]...)
    return vsp
end

"""
    randFiniteMPS(elt, state, flux; side=:right, normalize=true)

Create a random finite MPS in the charged sector obtained by adding `flux` to `state`.
The internal bond spaces are capped by the bond-space profile of `state`.

Use `side=:right` when the extra charge belongs to the right boundary and
`side=:left` when it belongs to the left boundary.
"""
function randFiniteMPS(
        elt::Type{<:Number}, state::FiniteNormalMPS, flux;
        side::Symbol = :right, normalize::Bool = true
    )
    L = length(state)
    maxvspaces = Vector{typeof(left_virtualspace(state, 1))}(undef, L - 1)
    for i in 1:(L - 1)
        maxvspaces[i] = right_virtualspace(state, i)
    end

    left = left_virtualspace(state, 1)
    right = right_virtualspace(state, L)
    if side === :right
        right = fuse(right, flux)
    elseif side === :left
        left = fuse(left, flux)
    else
        throw(ArgumentError("invalid side :$side, expected :right or :left"))
    end

    return FiniteMPS(
        rand, elt, physicalspace(state), maxvspaces;
        left = left, right = right, normalize = normalize
    )
end


"""
    randFiniteMPS(elt, state::FiniteNormalMPS, operator::AbstractTensorMap)

Create a random finite MPS in the charge sector selected by `operator`.

For `(1,2)` operators the extra virtual space is taken from the second domain
leg and attached to the right boundary. For `(2,1)` operators it is taken from
the first codomain leg and attached to the left boundary. Charge-neutral
`(1,1)` operators keep the reference boundary sector.
"""
function randFiniteMPS(elt::Type{<:Number}, state::FiniteNormalMPS, operator::AbstractTensorMap)
    if (length(domain(operator)) == 2)&&(length(codomain(operator)) == 1)
        vspace = domain(operator)[2]
        return randFiniteMPS(elt, state, vspace)
    elseif (length(codomain(operator)) == 2)&&(length(domain(operator)) == 1)
        vspace = codomain(operator)[1]
        return randFiniteMPS(elt, state, vspace; side=:left)
    elseif (length(codomain(operator)) == 1)&&(length(domain(operator)) == 1)
        vspace = oneunit(right_virtualspace(state, length(state)))
        return randFiniteMPS(elt, state, vspace)
    else
        throw(ArgumentError("invalid operator, expected 2-leg or 3-leg tensor"))
    end
end
