using QuantumLattices: Hilbert,Fock, Term, Hopping, Hubbard, Lattice, Neighbors, bonds, OperatorGenerator, Operator, CompositeIndex
using TensorKit
using MPSKit
using DynamicalCorrelators
using MPSKitModels: contract_onesite, contract_twosite, FiniteStrip, FiniteCylinder, FiniteChain

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

@testset "charged state" begin
    elt = Float64
    filling = (1, 1)
    L=4
    @testset "U1" begin
        H = hubbard(elt, U1Irrep, U1Irrep, FiniteChain(L); filling=filling, t=1, U=8, mu=0)
        st = randFiniteMPS(elt, U1Irrep, U1Irrep, L; filling=filling)
        gs, envs, delta = find_groundstate(st, H, DMRG2(trscheme = truncerror(; rtol=1e-10)));
        ep =  e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
        sp = S_plus(elt, U1Irrep, U1Irrep; side=:L, filling=filling)
        sz = S_z(elt, U1Irrep, U1Irrep; filling=filling)
        i, j = 1, 4
        cgs₁ = chargedMPS(ep, gs, i)
        cgs₂ = chargedMPS(ep, gs, j)
        @test isapprox(dot(cgs₁, cgs₂), 0.02313258689229983; atol=1e-5)
        sgs₁ = chargedMPS(sp, gs, i)
        sgs₂ = chargedMPS(sp, gs, j)
        @test isapprox(dot(sgs₁, sgs₂), -0.1556937701438006; atol=1e-5)
        sgs₃ = chargedMPS(sz, gs, i)
        sgs₄ = chargedMPS(sz, gs, j)
        @test isapprox(dot(sgs₃, sgs₄), -0.07784688507190048; atol=1e-5)
    end
    @testset "SU2" begin
        H = hubbard(elt, SU2Irrep, U1Irrep, FiniteChain(4); filling=filling, t=1, U=8, mu=0)
        st = randFiniteMPS(elt, SU2Irrep, U1Irrep, 4; filling=filling)
        gs, envs, delta = find_groundstate(st, H, DMRG2(trscheme = truncerror(; rtol=1e-6)));
        ep = e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
        sp = S_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
        i, j = 1, 4
        cgs₁ = chargedMPS(ep, gs, i)
        cgs₂ = chargedMPS(ep, gs, j)
        @test isapprox(dot(cgs₁, cgs₂), 0.02313258689229983+0.02313258689229991; atol=1e-5)
        sgs₁ = chargedMPS(sp, gs, i)
        sgs₂ = chargedMPS(sp, gs, j)
        @test isapprox(dot(sgs₁, sgs₂), (-0.1556937701438006-0.1556937701438008)/2-0.07784688507190048; atol=1e-5)
    end
end
