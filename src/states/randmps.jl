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

function randFiniteMPS(elt::Type{<:Number}, H::MPOHamiltonian; L=10, right=oneunit(physicalspace(H)[1]))
    Ps = physicalspace(H)
    Vs = restrict_virtualspaces(Ps; right=right, L=L)
    FiniteMPS(rand, elt, Ps, Vs[2:(end - 1)]; right=right)
end

function randFiniteMPS(elt::Type{<:Number}, pspace, N::Integer; right=oneunit(pspace))
    pspaces = repeat([pspace], N)
    vspaces = restrict_virtualspaces(pspaces; right=right)
    FiniteMPS(rand, elt, pspaces, vspaces[2:(end - 1)]; right=right)
end

function restrict_virtualspaces(Ps; left=oneunit(Ps[1]), right=oneunit(Ps[1]), L=10)
    if length(Ps) <= L
        Vs = max_virtualspaces(Ps; right=right)
    else
        Vs = similar(Ps, length(Ps) + 1)
        Vs[1] = left
        Vs[end] = right
        for k in 2:(L÷2)
            Vs[k] = fuse(Vs[k - 1], fuse(Ps[k - 1]))
        end
        for k in (L÷2+1):2:length(Ps)
            Vs[k] = Vs[4]
        end
        for k in (L÷2+2):2:(length(Ps)-1)
            Vs[k] = Vs[5]
        end 
        for k in reverse(2:length(Ps))
            Vs[k] = infimum(Vs[k], fuse(Vs[k + 1], dual(fuse(Ps[k]))))
        end
    end
    return Vs
end