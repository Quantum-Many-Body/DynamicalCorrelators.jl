function randFiniteMPS(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}, N::Integer; filling=(1,1), md=10, mz=nothing)
    P,Q = filling
    isodd(P)&&(mod(N, 2Q)==0) ? (k = 0:2Q) : iseven(P)&&(mod(N, 2Q)==0) ? (k = 0:Q) : throw(ArgumentError("invalid length for the filling"))
    mz == nothing ? (ℤ = -max(P,Q):max(P,Q)) : (ℤ = -mz:mz)
    I = FermionParity ⊠ U1Irrep ⊠ U1Irrep
    Vs = [_vspaces(U1Irrep, U1Irrep, P, Q, k[i], ℤ, I, md) for i in 2:length(k)]
    Ps = Vect[I]((0,0,-P) => 1, (0,0,2*Q-P) => 1, (1,1,Q-P) => 1, (1,-1,Q-P) => 1)
    pspaces = repeat([Ps,], N)
    M = div(N, 2Q)
    M == 1 ? maxvspaces = Vs[1:end-1] : maxvspaces = repeat(Vs, M)[1:end-1]
    randmps = FiniteMPS(rand, elt, pspaces, maxvspaces)
    return randmps
end

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
    M = div(N, 2Q)
    M == 1 ? maxvspaces = Vs[1:end-1] : maxvspaces = repeat(Vs, M)[1:end-1]
    randmps = FiniteMPS(rand, elt, pspaces, maxvspaces)
    return randmps
end

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