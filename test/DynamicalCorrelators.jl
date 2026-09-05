using Test
using TensorKit
using TensorKit: ⊠
using MPSKit
using DynamicalCorrelators
using MPSKitModels: contract_onesite, contract_twosite, FiniteStrip, FiniteCylinder, FiniteChain
using QuantumLattices
using ExactDiagonalization
using Printf

const ED_TOL_ENERGY = 1e-12
const ED_TOL_OBS = 1e-10

@testset "operators" begin
    elt = Float64
    for filling in [(1,2), (1,1), (3,2)]
        @testset "U1×U1 fermions" begin
            c⁺ul = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
            cur = e_min(elt, U1Irrep, U1Irrep; side=:R, spin=:up, filling=filling)
            c⁺dl = e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
            cdr = e_min(elt, U1Irrep, U1Irrep; side=:R, spin=:down, filling=filling)
            cul = e_min(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
            c⁺ur = e_plus(elt, U1Irrep, U1Irrep; side=:R, spin=:up, filling=filling)
            cdl = e_min(elt, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
            c⁺dr = e_plus(elt, U1Irrep, U1Irrep; side=:R, spin=:down,filling=filling)
            @test (contract_onesite(c⁺ul, cur) - contract_onesite(cul, c⁺ur)) == isomorphism(codomain(contract_onesite(c⁺ul, cur)), domain(contract_onesite(c⁺ul, cur)))
            @test (contract_onesite(c⁺dl, cdr) - contract_onesite(cdl, c⁺dr)) == isomorphism(codomain(contract_onesite(c⁺dl, cdr)), domain(contract_onesite(c⁺dl, cdr)))
            @test number(elt, U1Irrep, U1Irrep; filling=filling) == contract_onesite(c⁺ul, cur) + contract_onesite(c⁺dl, cdr)
            @test onsiteCoulomb(elt, U1Irrep, U1Irrep; filling=filling) ≈ contract_onesite(contract_onesite(c⁺ul, cur), contract_onesite(c⁺dl, cdr))
        end
        @testset "U1×U1 spin operators" begin
            s⁺l = S_plus(elt, U1Irrep, U1Irrep; side=:L, filling=filling)
            sr = S_min(elt, U1Irrep, U1Irrep; side=:R, filling=filling)
            sl = S_min(elt, U1Irrep, U1Irrep; side=:L, filling=filling)
            s⁺r = S_plus(elt, U1Irrep, U1Irrep; side=:R, filling=filling)
            sz = S_z(elt, U1Irrep, U1Irrep; filling=filling)
            nbc1 = neiborCoulomb(elt, U1Irrep, U1Irrep, true; filling=filling)
            nbc2 = neiborCoulomb(elt, U1Irrep, U1Irrep, false; filling=filling)
            sf = spinflip(elt, U1Irrep, U1Irrep; filling=filling)
            SS1 = -nbc1/4 + nbc2/4 + sf/2
            SS2 = heisenberg(elt, U1Irrep, U1Irrep; filling=filling)
            @test (contract_onesite(s⁺l, sr) - contract_onesite(sl, s⁺r)) ≈ 2*sz
            @test SS1==SS2
        end
        @testset "SU2×U1 fermions" begin
            c⁺l = e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
            cr = e_min(elt, SU2Irrep, U1Irrep; side=:R, filling=filling)
            cl = e_min(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
            c⁺r = e_plus(elt, SU2Irrep, U1Irrep; side=:R, filling=filling)
            @test (contract_onesite(c⁺l, cr) - contract_onesite(cl, c⁺r)) ≈ 2*isomorphism(codomain(contract_onesite(c⁺l, cr)), domain(contract_onesite(c⁺l, cr)))
            @test number(elt, SU2Irrep, U1Irrep; filling=filling) ≈ contract_onesite(c⁺l, cr)
            @test onsiteCoulomb(elt, SU2Irrep, U1Irrep; filling=filling) ≈ (contract_onesite(contract_onesite(c⁺l, cr), contract_onesite(c⁺l, cr)) - contract_onesite(c⁺l, cr))/2
        end
        @testset "SU2×U1 spin operators" begin
            s⁺l = S_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
            sr = S_min(elt, SU2Irrep, U1Irrep; side=:R, filling=filling)
            sq = S_square(elt, SU2Irrep, U1Irrep; filling=filling)
            @test contract_onesite(s⁺l, sr) ≈ sq
        end
    end
end

@testset "Hamiltonian" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1, 0], [0, 1]])
    lattice₁ = Lattice(unitcell, (2, 2), ('o', 'o'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(lattice₁))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    H₁ = hamiltonian((t, U), lattice₁, hilbert; neighbors=1)
    H₂ = hubbard(Float64, U1Irrep, U1Irrep, FiniteStrip(2, 4); t=1.0, U=8.0, mu=0.0, filling=(1,1))
    @test H₁ ≈ H₂
    lattice₂ = Lattice(unitcell, (2, 2), ('p', 'o'))
    H₃ = hamiltonian((t, U), lattice₂, hilbert; neighbors=1)
    H₄ = hubbard(Float64, U1Irrep, U1Irrep, FiniteCylinder(2, 4); t=1.0, U=8.0, mu=0.0, filling=(1,1))
    @test H₃ ≈ H₄
end

function G_ed_SU2(gs, F, L, hilbert)
    G = zeros(ComplexF64, L, L)
    for i in 1:L, j in 1:L
        opt = Operator(1, 𝕔(j, 1, -1//2))+Operator(1, 𝕔(j, 1, 1//2))
        opm = matrix(opt,(BinaryBases(2*L, ℕ(F-1)),BinaryBases(2*L,ℕ(F))),Table(hilbert, Metric(EDKind(hilbert), hilbert)))
        psi1 = opm*gs

        opt = Operator(1, 𝕔(i, 1, -1//2))+Operator(1, 𝕔(i, 1, 1//2))
        opm = matrix(opt,(BinaryBases(2*L, ℕ(F-1)),BinaryBases(2*L,ℕ(F))),Table(hilbert, Metric(EDKind(hilbert), hilbert)))
        psi2 = opm*gs
        G[i, j] = dot(psi2, psi1)
    end
    return G
end

function pairing_N(gs, F, L, (i,j,k,l), hilbert)
    opt = (1/sqrt(2))*(Operator(1, 𝕔(i, 1, -1//2))*Operator(1, 𝕔(j, 1, 1//2))-Operator(1, 𝕔(i, 1, 1//2))*Operator(1, 𝕔(j, 1, -1//2)))
    opm = matrix(opt,(BinaryBases(2*L, ℕ(F-2)),BinaryBases(2*L,ℕ(F))),Table(hilbert, Metric(EDKind(hilbert), hilbert)))
    psi1 = opm*gs
    opt = (1/sqrt(2))*(Operator(1, 𝕔(k, 1, -1//2))*Operator(1, 𝕔(l, 1, 1//2))-Operator(1, 𝕔(k, 1, 1//2))*Operator(1, 𝕔(l, 1, -1//2)))
    opm = matrix(opt,(BinaryBases(2*L, ℕ(F-2)),BinaryBases(2*L,ℕ(F))),Table(hilbert, Metric(EDKind(hilbert), hilbert)))
    psi2 = opm*gs
    dot(psi1, psi2)
end

function G_ed_Sz(gs, F, L, hilbert; sw=1:8, su=9:16)
    G = zeros(ComplexF64, L, L)
    for i in 1:L, j in 1:L
        oi1 = Operator(1, 𝕔(i, 1, 1//2))
        oi2 = Operator(1, 𝕔(i, 1, -1//2))
        oj1 = Operator(1, 𝕔(j, 1, 1//2))
        oj2 = Operator(1, 𝕔(j, 1, -1//2))
        pj1 = matrix(oj1,(BinaryBases(sw, su, 𝕊ᶻ(F-1/2)), BinaryBases(sw, su, 𝕊ᶻ(F))),Table(hilbert, Metric(EDKind(hilbert), hilbert)))*gs
        pj2 = matrix(oj2,(BinaryBases(sw, su, 𝕊ᶻ(F+1/2)), BinaryBases(sw, su, 𝕊ᶻ(F))),Table(hilbert, Metric(EDKind(hilbert), hilbert)))*gs
        pi1 = matrix(oi1,(BinaryBases(sw, su, 𝕊ᶻ(F-1/2)), BinaryBases(sw, su, 𝕊ᶻ(F))),Table(hilbert, Metric(EDKind(hilbert), hilbert)))*gs
        pi2 = matrix(oi2,(BinaryBases(sw, su, 𝕊ᶻ(F+1/2)), BinaryBases(sw, su, 𝕊ᶻ(F))),Table(hilbert, Metric(EDKind(hilbert), hilbert)))*gs
        G[i, j] = dot(pi1, pj1) + dot(pi2, pj2)
    end
    return G
end

function pairing_Sz(gs, F, (i,j,k,l), hilbert; sw=1:8, su=9:16)
    opt = (1/sqrt(2))*(Operator(1, 𝕔(i, 1, -1//2))*Operator(1, 𝕔(j, 1, 1//2))-Operator(1, 𝕔(i, 1, 1//2))*Operator(1, 𝕔(j, 1, -1//2)))
    opm = matrix(opt,(BinaryBases(sw, su, 𝕊ᶻ(F)), BinaryBases(sw, su, 𝕊ᶻ(F))),Table(hilbert, Metric(EDKind(hilbert), hilbert)))
    psi1 = opm*gs
    opt = (1/sqrt(2))*(Operator(1, 𝕔(k, 1, -1//2))*Operator(1, 𝕔(l, 1, 1//2))-Operator(1, 𝕔(k, 1, 1//2))*Operator(1, 𝕔(l, 1, -1//2)))
    opm = matrix(opt,(BinaryBases(sw, su, 𝕊ᶻ(F)), BinaryBases(sw, su, 𝕊ᶻ(F))),Table(hilbert, Metric(EDKind(hilbert), hilbert)))
    psi2 = opm*gs
    dot(psi1, psi2)
end

function dcorrelator_ed(t, lattice, hilbert, term)
    bs = BinaryBases(8, ℕ(4))
    bs₁ =  BinaryBases(8, ℕ(3))
    bs₂ =  BinaryBases(8, ℕ(5))
    ops = OperatorGenerator(bonds(lattice, 1), hilbert, term)
    table = Table(hilbert, Metric(EDKind(hilbert), hilbert))
    Hₘ = matrix(expand(ops), (bs, bs), table) 
    vals, vecs = eigen(Matrix(Hₘ))
    Hₘ2 = matrix(expand(ops), (bs₂, bs₂), table) 
    vals2, vecs2 = eigen(Matrix(Hₘ2))
    id = sort(collect(keys(table)), by = x -> table[x])
    ops₁, ops₂ = [Operator(1, 𝕔(key[2], key[3], key[1])) for key in id], [Operator(1, 𝕔⁺(key[2], key[3], key[1])) for key in id] 
    ops₁, ops₂ = [ops₁[i] + ops₁[i+length(ops₁)÷2] for i in 1:length(ops₁)÷2], [ops₂[i] + ops₂[i+length(ops₂)÷2] for i in 1:length(ops₂)÷2]
    opm₁, opm₂ = [matrix(op, (bs₁, bs), table) for op in ops₁], [matrix(op, (bs₂, bs), table) for op in ops₂]
    ccd = zeros(ComplexF64, length(opm₂), length(opm₂))
    for i in 1:length(opm₂)
        for j in 1:length(opm₂)
            for m in 1:length(vals2)
                ccd[i, j] += exp(im*vals[1]*t)*exp(-vals2[m]*t*im)*conj(dot(vecs2[:,m],opm₂[i]*vecs[:,1]))*dot(vecs2[:,m],opm₂[j]*vecs[:,1])
            end
        end
    end
    return -im*ccd
end

function trrho(beta, lattice, hilbert, term)
    bs = BinaryBases(8)
    ops = OperatorGenerator(bonds(lattice, 1), hilbert, term)
    table = Table(hilbert, OperatorIndexToTuple(:spin, :site, :orbital))
    Hₘ = matrix(expand(ops), (bs, bs), table) 
    vals, vecs = eigen(Matrix(Hₘ))
    Z = 0
    for n in 1:length(vals)
        Z += exp(-vals[n]*beta)
    end
    return Z
end

@testset "ED benchmark" begin
    elt = ComplexF64
    N = 8
    L = 8
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
    lattice = Lattice(unitcell, (8, ), ('o',))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(lattice))
    t = Hopping(:t, ComplexF64(-1.0), 1; amplitude = b->b.points[1].site < b.points[2].site ? 1+0.5im : 1-0.5im)
    U = Hubbard(:U, 8.0)

    @testset "spin-SU2 x particle-U1" begin
        filling = (1, 2)

        # DMRG side (test_gs.jl)
        st = randFiniteMPS(elt, SU2Irrep, U1Irrep, N; filling=filling)
        truncdims = [64, 64, 128, 128, 128, 128]
        H = hubbard(elt, SU2Irrep, U1Irrep, Custom(lattice); t=-1-0.5im, U=8, filling=filling)
        gs, env, E0 = dmrg2!(st, H, truncdims; save=false, verbose=false)

        cm = e_min(elt, SU2Irrep, U1Irrep; filling=filling)
        gf = zeros(ComplexF64, 8, 8)
        for j in 1:8
                gf[:, j] = sweep_dot(gs, cm, chargedMPS(cm, gs, j))
        end

        sd = singlet_dagger(elt, SU2Irrep, U1Irrep; filling=filling)
        sl  = singlet(elt, SU2Irrep, U1Irrep; filling=filling, side=:R)
        i = 1
        j = 2
        k = 2
        l = 4
        pc = correlator(gs, sd, sl, (i,j), (k,l))

        quantumnumber = ℕ(4)
        ed = ED(lattice, hilbert, (t, U), quantumnumber)
        eigensystem = eigen(ed; nev=1)
        gs_ed = eigensystem.vectors[1]
        E_ed = eigensystem.values[1]
        G = G_ed_SU2(gs_ed, 4, 8, hilbert)
        pc_ed = pairing_N(gs_ed, 4, 8, (i,j,k,l), hilbert)

        @printf("SU2xU1: E_dmrg = %.12f, E_ed = %.12f, |ΔE| = %.2e\n", real(E0), real(E_ed), abs(E0 - E_ed))
        @printf("        max|ΔG| = %.2e, |Δpc| = %.2e\n", maximum(abs.(gf - G)), abs(pc - pc_ed))
        @test isapprox(E0, E_ed; atol = ED_TOL_ENERGY)
        @test isapprox(gf, G; atol = ED_TOL_OBS)
        @test isapprox(pc, pc_ed; atol = ED_TOL_OBS)
    end

    @testset "spin-SU2 only, Sz = 1/2 sector" begin
        H = hubbard(elt, SU2Irrep, Custom(lattice); t=-1.0-0.5im, U=8.0)
        vs = repeat([Vect[(FermionParity ⊠ SU2Irrep)]((Int(!isinteger(i)), i) => 1  for i in 0:1//2:1),], L - 1)
        st = FiniteMPS(rand, elt, physicalspace(H), vs; right=Vect[(FermionParity ⊠ SU2Irrep)]((1, 1/2) => 1))
        truncs = [64, 64, 128, 128, 128, 128]
        gs, env, E0 = dmrg2!(st, H, truncs; save=false, verbose=false)

        cm = e_min(elt, SU2Irrep)
        gf = zeros(ComplexF64, 8, 8)
        for j in 1:8
                gf[:, j] = sweep_dot(gs, cm, chargedMPS(cm, gs, j))
        end

        sd = singlet_dagger(elt, SU2Irrep)
        sl  = singlet(elt, SU2Irrep; side=:R)
        i = 3
        j = 6
        k = 3
        l = 5
        pc = correlator(gs, sd, sl, (i,j), (k,l))

        quantumnumber = 𝕊ᶻ(1//2)
        ed = ED(lattice, hilbert, (t, U), quantumnumber)
        eigensystem = eigen(ed; nev=1)
        gs_ed = eigensystem.vectors[1]
        E_ed = eigensystem.values[1]
        G = G_ed_Sz(gs_ed, 1//2, 8, hilbert)
        pc_ed = pairing_Sz(gs_ed, 1//2, (i,j,k,l), hilbert)

        @printf("SU2:    E_dmrg = %.12f, E_ed = %.12f, |ΔE| = %.2e\n", real(E0), real(E_ed), abs(E0 - E_ed))
        @printf("        max|ΔG| = %.2e, |Δpc| = %.2e\n", maximum(abs.(gf - G)), abs(pc - pc_ed))
        @test isapprox(E0, E_ed; atol = ED_TOL_ENERGY)
        @test isapprox(gf, G; atol = ED_TOL_OBS)
        @test isapprox(pc, pc_ed; atol = ED_TOL_OBS)
    end
    
    @testset "Dynamical Green's function" begin
        lattice = Lattice(unitcell, (4, ), ('o',))
        hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(lattice))
        gf_ed = dcorrelator_ed(1.0, lattice, hilbert, (t, U))

        filling = (1, 1)
        st = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, 4; filling=filling)
        truncdims = [64, 64, 64, 64]
        H = hubbard(ComplexF64, SU2Irrep, U1Irrep, Custom(lattice);t=-1-0.5im, U=8, filling=filling)
        gs, env ,E0 = dmrg2!(st, H, truncdims; save=false, verbose=false);
        cp = e_plus(ComplexF64, SU2Irrep, U1Irrep; filling=filling)
        gf_tdvp = dcorrelator(gs, H, cp, 1:4;
                            verbose=false,
                            save=false,
                            times=0:0.05:1.0,
                            n=3,
                            tdvp1 = myTDVP1_CBE(; D = 128),
                            tdvp2 = myTDVP2(; trunc=truncrank(128)),
                            )
        df = maximum(abs.((gf_tdvp[:,:,end].-gf_ed)))
        @printf("Dynamical Green's function: norm(G_tdvp-G_ed) = %.2e\n", df)
        @test  df < 1e-4
    end

    @testset "Finite temperature" begin
        lattice = Lattice(unitcell, (4, ), ('o',))
        hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(lattice))
        Z_ed = trrho(2.0, lattice, hilbert, (t, U))# β = 2.0
        H = hubbard(ComplexF64, SU2Irrep, U1Irrep, Custom(lattice);t=-1-0.5im, U=8)
        bs = 0:0.05:1.0 # max bs = β/2
        rho = evolve_mps(H, -im*bs; 
                            tdvp1 = myTDVP1_CBE(;D=128),
                            tdvp2 = myTDVP2(; trunc=truncrank(128)),
                            verbose=false,
                            save=false)
        Z_tdvp = dot(rho, rho)
        @printf("Finite temperature: Z_tdvp = %.12f, Z_ed = %.12f, |ΔZ| = %.2e\n", real(Z_tdvp), real(Z_ed), abs(Z_tdvp-Z_ed))
        @test isapprox(real(Z_tdvp), real(Z_ed); atol = 1e-4)
    end

end
