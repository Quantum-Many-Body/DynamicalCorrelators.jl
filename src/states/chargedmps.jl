const FiniteNormalMPS{C} = FiniteMPS{<:AbstractTensorMap{N,C,2,1}} where {N}
const FiniteSuperMPS{C} = FiniteMPS{<:AbstractTensorMap{N,C,3,1}} where {N}

"""
    chargedMPS(operator::AbstractTensorMap, state::FiniteNormalMPS, site::Integer)
"""
function chargedMPS(operator::AbstractTensorMap, state::FiniteNormalMPS, site::Integer)
    return chargedMPO(operator, site, length(state))*state
end

function chargedMPS(operator::AbstractTensorMap, state::FiniteNormalMPS, site₁::Integer, site₂::Integer)
    return chargedMPO(operator, site₁, site₂, length(state))*state
end

"""
    chargedMPS(op::AbstractTensorMap{B,S,1,2}, mps::FiniteSuperMPS, site::Integer) where {B, S}
"""
function chargedMPS(op::AbstractTensorMap{B,S,1,2}, mps::FiniteSuperMPS, site::Integer) where {B, S}
    T = promote_contract(scalartype(op), scalartype(mps))
    A = similarstoragetype(eltype(mps), T)
    A2 = map(1:length(mps)) do i
        A1 = i == 1 ? mps.AC[1] : mps.AR[i]
        if i < site
            a = A1
        end
        if i == site 
            F = fuser(A, domain(A1, 1), domain(op, 2))
            @plansor a[-1 -2 -3; -4] := A1[-1 1 3; 5] * op[-2; 1 2] * τ[2 -3; 3 4] * conj(F[-4; 5 4])
        end
        if i > site
            Fl, Fr = fuser(A, codomain(A1, 1), domain(op, 2)), fuser(A, domain(A1, 1), domain(op, 2))
            @plansor a[-1 -2 -3; -4] := Fl[-1; 1 2] * A1[1 3 5; 7] * τ[2 -2; 3 4] * τ[4 -3; 5 6] * conj(Fr[-4; 7 6])
        end
        return a
    end
    trscheme = truncbelow(eps(real(T)))
    return changebonds!(FiniteMPS(A2), SvdCut(; trscheme); normalize = false)
end

"""
chargedMPS(op::AbstractTensorMap{S,B,2,1}, mps::FiniteSuperMPS, site::Integer) where {B, S}
"""
function chargedMPS(op::AbstractTensorMap{S,B,2,1}, mps::FiniteSuperMPS, site::Integer) where {B, S}
    T = promote_contract(scalartype(op), scalartype(mps))
    A = similarstoragetype(eltype(mps), T)
    A2 = map(1:length(mps)) do i
        A1 = i == 1 ? mps.AC[1] : mps.AR[i]
        if i < site
            Fl, Fr = fuser(A, codomain(A1, 1), codomain(op, 1)), fuser(A, domain(A1, 1), codomain(op, 1))
            @plansor a[-1 -2 -3; -4] := Fl[-1; 1 2] * A1[1 3 5; 7] * τ[2 -2; 3 4] * τ[4 -3; 5 6] * conj(Fr[-4; 7 6])
        end
        if i == site 
            F = fuser(A, codomain(A1, 1), codomain(op, 1))
            @plansor a[-1 -2 -3; -4] := F[-1; 1 2] * A1[1 3 -3; -4] * op[2 -2; 3]
        end
        if i > site
            a = A1
        end
        return a
    end
    trscheme = truncbelow(eps(real(T)))
    return changebonds!(FiniteMPS(A2), SvdCut(; trscheme); normalize = false)
end

"""
    chargedMPS(op::AbstractTensorMap{S,B,1,1}, mps::FiniteSuperMPS, site::Integer) where {B, S}
"""
function chargedMPS(op::AbstractTensorMap{S,B,1,1}, mps::FiniteSuperMPS, site::Integer) where {B, S}
    T = promote_contract(scalartype(op), scalartype(mps))
    A2 = map(1:length(mps)) do i
        A1 = i == 1 ? mps.AC[1] : mps.AR[i]
        if i !== site
            a = A1
        else
            @plansor a[-1 -2 -3; -4] := A1[-1 1 -3; -4] * op[-2; 1]
        end
        return a
    end
    trscheme = truncbelow(eps(real(T)))
    return changebonds!(FiniteMPS(A2), SvdCut(; trscheme); normalize = false)
end

function chargedMPS(op::AbstractTensorMap, gs::AbstractFiniteMPS; tol=1e-6, maxiter=30, krylovdim=8, trscheme=truncdim(dims(domain(gs[length(gs)÷2]))[1]), cgs_path::String="./")
    !isdir(cgs_path)&& mkdir(cgs_path)
    filename = joinpath(gf_path, "chargedMPS_N=$(length(gs)).jld2")
    jldopen(filename, "w") do f
        f["gs"] = gs
    end
    alg = DMRG2(; tol=tol, maxiter=maxiter, verbosity=3,
            alg_eigsolve= Lanczos(;
                krylovdim = krylovdim,
                maxiter = 1,
                tol = 1e-8,
                orth = ModifiedGramSchmidt(),
                eager = false,
                verbosity = 0), 
            alg_svd= SDD(), 
            trscheme=trscheme)
    record_start = now()
    println("chargedMPS start", Dates.format(start_time, "d.u yyyy HH:MM"))
    @sync @distributed for i in 1:length(gs)
        ags, _, ϵ = approximate(cgs[i], (chargedMPO(op, i, length(gs)), gs), alg)
        jldopen(filename, "a") do f
            f["cgs_$(i)"] = ags
            f["eps_$(i)"] = ϵ
        end
        println("chargedMPS_$(i) is finished, ϵ=$(ϵ)", Dates.canonicalize(now()-record_start))
    end
    GC.gc()
    return nothing
end

"""
    identityMPS(H::FiniteMPOHamiltonian)
"""
function identityMPS(H::FiniteMPOHamiltonian)
    V = oneunit(spacetype(H))
    W = map(1:length(H)) do site
        return BraidingTensor{scalartype(H)}(physicalspace(H, site), V)
    end
    return convert(FiniteMPS, FiniteMPO(W))
end