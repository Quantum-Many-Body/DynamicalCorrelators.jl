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
    randFiniteMPS(elt, H::MPOHamiltonian; left=oneunit(...), right=oneunit(...))

Create a random finite MPS compatible with the given MPO Hamiltonian `H`.
The virtual spaces are automatically inferred from the physical spaces and boundary conditions.
"""
function randFiniteMPS(elt::Type{<:Number}, H::MPOHamiltonian; left=oneunit(physicalspace(H)[1]), right=oneunit(physicalspace(H)[1]))
    Ps = physicalspace(H)
    temp = restrict_virtualspaces(Ps; left=left, right=right)
    st = FiniteMPS(rand, elt, Ps,temp[2:(end - 1)]; left=left, right=right)
    changebonds!(st, SvdCut(;trscheme=truncrank(32)))
    Vs = Vector{eltype(Ps)}(undef, length(temp))
    for i in 1:length(Ps)
        Vs[i] = left_virtualspace(st[i])
    end
    Vs[length(Ps)+1] = right_virtualspace(st[length(Ps)])
    return FiniteMPS(rand, elt, Ps, Vs[2:end-1]; left=left, right=right)
end


"""
    randFiniteMPS(elt, pspace, N; right=oneunit(pspace))

Create a random finite MPS with `N` sites, each having physical space `pspace`.
"""
function randFiniteMPS(elt::Type{<:Number}, pspace, N::Integer; right=oneunit(pspace))
    pspaces = repeat([pspace], N)
    vspaces = restrict_virtualspaces(pspaces; right=right)
    FiniteMPS(rand, elt, pspaces, vspaces[2:(end - 1)]; right=right)
end

"""
    restrict_virtualspaces(Ps; left=oneunit(Ps[1]), right=oneunit(Ps[1]))

Compute virtual spaces for a chain of physical spaces `Ps`.
For short chains (N ≤ 12), uses exact `max_virtualspaces`.
For longer chains, uses an interpolation scheme from a 10-site reference to avoid
exponential growth of quantum number sectors.
"""
function restrict_virtualspaces(Ps; left=oneunit(Ps[1]), right=oneunit(Ps[1]))
    N = length(Ps)
    if  N <= 12
        Vs = max_virtualspaces(Ps; left=left, right=right)
    else
        Vs = Vector{eltype(Ps)}(undef, N+1)
        temp = max_virtualspaces(Ps[1:10]; left=left, right=right)
        Vs[1:3] = temp[1:3]
        Vs[N-1:N+1] = temp[9:11]
        Vs[N÷2+1] = temp[6]
        Vs[4:2:(N÷2-1)] = [temp[4] for _ in 1:length(4:2:(N÷2-1))]
        Vs[5:2:N÷2] = [temp[5]  for _ in 1:length(5:2:N÷2)]
        Vs[(N÷2+2):2:(N-3)] = [temp[7] for _ in 1:length((N÷2+2):2:(N-3))]
        Vs[(N÷2+3):2:(N-2)] = [temp[8] for _ in 1:length((N÷2+3):2:(N-2))]
    end
    return Vs
end