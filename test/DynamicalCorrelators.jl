using Test
using LinearAlgebra
using TensorKit
using TensorKit: ⊠
using MPSKit
using DynamicalCorrelators
using MPSKitModels: contract_onesite, contract_twosite, FiniteStrip, FiniteCylinder, FiniteChain
using QuantumLattices
using ExactDiagonalization
using Printf

@testset "MomentumMPS initial state" begin
    # The public initializer constructs the A14 source and its right background
    # internally from the ground-state MPS and a local operator.
    gs = InfiniteMPS(rand, ComplexF64, [ℂ^2], [ℂ^2])
    O = isomorphism(ComplexF64, ℂ^2, ℂ^2)

    ϕ = qmps(gs, O, 0.73)
    @test ϕ isa MomentumMPS
    @test length(ϕ) == 1
    @test length(ϕ.windowmps[1]) == 2
    @test check_momentum_mps(ϕ)
    @test isapprox(norm(ϕ), norm(ϕ.windowmps[1]); atol=1e-10, rtol=1e-10)

    # The wrapper enforces the q-MPS gauge and the initializer currently
    # implements the paper's two-site linear projection at this stage.
    @test_throws ArgumentError MomentumMPS(WindowMPS(gs, [gs.AR[1], gs.AR[1]], gs), 0.73)
    @test_throws ArgumentError qmps(gs, O, 0.0)
    @test_throws ArgumentError qmps(gs, O, complex(0.73, 0.1))
end

@testset "MomentumMPS multi-site unit cell" begin
    gs = InfiniteMPS(rand, ComplexF64, [ℂ^2, ℂ^2], [ℂ^2, ℂ^2])
    O = isomorphism(ComplexF64, ℂ^2, ℂ^2)

    ϕ = qmps(gs, O, 0.37)
    @test ϕ isa MomentumMPS
    @test length(ϕ) == 2
    @test map(length, ϕ.windowmps) == [2, 2]
    @test check_momentum_mps(ϕ)
    @test ϕ.windowmps[1].left_gs.AL == gs.AL
    @test ϕ.windowmps[2].left_gs.AL == circshift(gs, 1).AL
    @test ϕ.windowmps[1].right_gs.AL == gs.AL
    @test ϕ.windowmps[2].right_gs.AL == circshift(gs, 1).AL
end

@testset "MomentumMPS symmetry sectors" begin
    @testset "U1 x U1 right-charged source" begin
        I = FermionParity ⊠ U1Irrep ⊠ U1Irrep
        O = e_plus(ComplexF64, U1Irrep, U1Irrep; side=:L)
        pspace = domain(O)[1]
        bond = Vect[I]((0, 0, 0) => 1, (1, 1, 0) => 1)
        gs = InfiniteMPS(rand, ComplexF64, [pspace], [bond])

        ϕ = qmps(gs, O, 0.73)
        @test ϕ isa MomentumMPS
        @test check_momentum_mps(ϕ)
        @test right_virtualspace(ϕ.windowmps[1], length(ϕ.windowmps[1])) ==
            right_virtualspace(ϕ.windowmps[1].right_gs, length(ϕ.windowmps[1]))
    end

    @testset "U1 x U1 right-charged two-site cell" begin
        I = FermionParity ⊠ U1Irrep ⊠ U1Irrep
        O = e_plus(ComplexF64, U1Irrep, U1Irrep; side=:L)
        pspace = domain(O)[1]
        bond = Vect[I]((0, 0, 0) => 1, (1, 1, 0) => 1)
        gs = InfiniteMPS(rand, ComplexF64, [pspace, pspace], [bond, bond])

        ϕ = qmps(gs, O, 0.37)
        @test length(ϕ) == 2
        @test check_momentum_mps(ϕ)
        charged = InfiniteMPO(fill(fZ(O), length(gs))) * gs
        for (i, window) in enumerate(ϕ.windowmps)
            shifted = circshift(charged, i - 1)
            @test window.right_gs.AL == shifted.AL
            @test right_virtualspace(window, length(window)) ==
                right_virtualspace(window.right_gs, length(window))
        end
    end

    @testset "SU2 x U1 right-charged source" begin
        I = FermionParity ⊠ SU2Irrep ⊠ U1Irrep
        O = e_plus(ComplexF64, SU2Irrep, U1Irrep; side=:L)
        pspace = domain(O)[1]
        bond = Vect[I]((0, 0, 0) => 1, (1, 1//2, 0) => 1)
        gs = InfiniteMPS(rand, ComplexF64, [pspace], [bond])

        ϕ = qmps(gs, O, 0.73)
        @test ϕ isa MomentumMPS
        @test check_momentum_mps(ϕ)
        @test right_virtualspace(ϕ.windowmps[1], length(ϕ.windowmps[1])) ==
            right_virtualspace(ϕ.windowmps[1].right_gs, length(ϕ.windowmps[1]))
    end
end

const ED_TOL_ENERGY = 1e-12
const ED_TOL_OBS = 1e-10
# The production default is a fixed one-step K=16 Lanczos.  The ED benchmark
# checks observables at 1e-10, so give its DMRG runs a larger Krylov budget
# while keeping the production default unchanged.
const ED_EIGSOLVE = DynamicalCorrelators._default_alg_eigsolve(
    24; tol=1e-12, eager=false
)

@testset "factorized CBE metadata and bidirectional update" begin
    Ly, Lx = 3, 4
    filling = (1, 1)
    coords = snake_2D([[0.0, 1.0], [sqrt(3) / 2, 1 / 2]],
        [o for j in 0:(Lx - 1) for o in (j < Lx - 1 ?
            [fill(iseven(j) ? 1 : -1, Ly - 1); 2] :
            fill(iseven(j) ? 1 : -1, Ly - 1))])
    lattice = Lattice(coords...; vectors = [[0.0, Float64(Ly)]])
    N = length(lattice)
    hilbert = Hilbert(site => Fock{:f}(1, 2) for site = 1:N)
    H = hamiltonian((Hopping(:t, -1.0, 1), Hubbard(:U, 8.0)), lattice, hilbert; filling)
    ψ = complex(randFiniteMPS(ComplexF64, U1Irrep, U1Irrep, N; filling))
    envs = HalfFiniteEnvironments(ψ, H)
    pos = N ÷ 2
    left, right = ψ.AC[pos], ψ.AR[pos + 1]
    left_c, _, _ = MPSKit.left_gauge(left)
    right_c = MPSKit._transpose_tail(right)
    GL, GR = MPSKit.leftenv(envs, pos, ψ), MPSKit.rightenv(envs, pos + 1, ψ)
    GR2 = GR[2:(end - 1)]
    channels, lA, lC, rA, rB, rows = DynamicalCorrelators._fc_channels(
        H[pos], H[pos + 1], GR2)
    @test !isempty(channels)
    @test !isempty(lA)
    @test !isempty(lC)
    @test !isempty(rA)
    @test !isempty(rB)
    @test sort!(unique!([a for m in channels for (a, _) in get(lA, m, [])])) == rows
    plan = DynamicalCorrelators._fc_graph(
        GL[2:(end - 1)], channels, lA, lC, rA, rB, rows,
        left, MPSKit._transpose_tail(right), left_c, right_c, nothing)

    # per-channel metadata: L nodes capture the inner-index-summed A1 terms
    # and the C1 block, R nodes the (b, A2, GR) terms and the B2 block
    np = DynamicalCorrelators.nonzero_pairs
    A1_of = Dict((I[1], I[4]) => b for (I, b) in np(H[pos].A))
    C1_of = Dict(I[3] => b for (I, b) in np(H[pos].C))
    A2_of = Dict((I[1], I[4]) => b for (I, b) in np(H[pos + 1].A))
    B2_of = Dict(I[1] => b for (I, b) in np(H[pos + 1].B))
    lnodes = filter(node -> node.kind == :L, plan.nodes)
    rnodes = filter(node -> node.kind == :R, plan.nodes)
    x2nodes = filter(node -> node.kind == :X2, plan.nodes)
    @test length(lnodes) == length(rnodes) == length(x2nodes) == length(channels)
    for (m, lnode, rnode) in zip(channels, lnodes, rnodes)
        for (a, A1) in getfield(lnode.apply, :aterms)
            @test A1 === A1_of[(a, m)]
        end
        C1b = getfield(lnode.apply, :C1b)
        C1b === nothing || @test C1b === C1_of[m]
        for (b, A2, GRb) in getfield(rnode.apply, :bterms)
            @test A2 === A2_of[(m, b)]
            @test GRb === GR2[b]
        end
        B2b = getfield(rnode.apply, :B2b)
        B2b === nothing || @test B2b === B2_of[m]
    end

    for rsvd in (false, true)
        alg = CBEExpand(; trunc = truncrank(2), rsvd)
        nrm = norm(ψ)
        Dr = dim(left_virtualspace(ψ, pos + 1))
        ψr = copy(ψ)
        envsr = HalfFiniteEnvironments(ψr, H)
        MPSKit.changebond!(pos, Val(:right), ψr, H, alg, envsr)
        @test norm(ψr) ≈ nrm
        @test dim(left_virtualspace(ψr, pos + 1)) == Dr + 2
        Dl = dim(left_virtualspace(ψ, pos))
        ψl = copy(ψ)
        envsl = HalfFiniteEnvironments(ψl, H)
        MPSKit.changebond!(pos, Val(:left), ψl, H, alg, envsl)
        @test norm(ψl) ≈ nrm
        @test dim(left_virtualspace(ψl, pos)) == Dl + 2
    end
end

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

    @testset "singlet pairing normalization" begin
        Δ = Pairing(:Δ, 0.16, 1, 𝕔𝕔(:, :, real(1im*σʸ));
            amplitude=bond -> real(exp(2im * azimuth(rcoordinate(bond)))))
        μ = Onsite(:μ, -1.2)
        ed = ED(lattice₁, hilbert, (t, μ, U, Δ), 𝕊ᶻ(0))
        eigensystem = eigen(ed; nev=1)
        E_ed = eigensystem.values[1]

        H = hubbard(ComplexF64, SU2Irrep, Custom(lattice₁);
            t=-1.0, U=8, mu=1.2, dw=0.16*sqrt(2))
        vs = repeat([Vect[(FermionParity ⊠ SU2Irrep)](
            (Int(!isinteger(i)), i) => 1 for i in 0:1//2:1),], length(H) - 1)
        st = FiniteMPS(rand, ComplexF64, physicalspace(H), vs)
        gs, env, E0 = dmrg2!(st, H, [64, 64, 128, 128];
            save=false, verbose=false, alg_eigsolve=ED_EIGSOLVE, etol=1e-12)
        @test isapprox(E0, E_ed; atol=ED_TOL_ENERGY)
    end
end

ed_c(i, σ) = Operator(1, 𝕔(i, 1, σ))
ed_cd(i, σ) = Operator(1, 𝕔⁺(i, 1, σ))
ed_sp(i) = ed_cd(i, 1//2) * ed_c(i, -1//2)
ed_sm(i) = ed_cd(i, -1//2) * ed_c(i, 1//2)
ed_sz(i) = (ed_cd(i, 1//2) * ed_c(i, 1//2) - ed_cd(i, -1//2) * ed_c(i, -1//2)) / 2

function ed_bond_singlet(i, j)
    return (ed_c(i, -1//2) * ed_c(j, 1//2) - ed_c(i, 1//2) * ed_c(j, -1//2)) / sqrt(2)
end
ed_onsite_singlet(i) = ed_bond_singlet(i, i)

function ed_spin_benchmark(ψ, sector, table, L)
    Czz = zeros(ComplexF64, L, L)
    Cpm = zeros(ComplexF64, L, L)
    Cmp = zeros(ComplexF64, L, L)
    for i in 1:L, j in 1:L
        Czz[i,j] = dot(ψ, matrix(ed_sz(i) * ed_sz(j), (sector, sector), table), ψ)
        Cpm[i,j] = dot(ψ, matrix(ed_sp(i) * ed_sm(j), (sector, sector), table), ψ)
        Cmp[i,j] = dot(ψ, matrix(ed_sm(i) * ed_sp(j), (sector, sector), table), ψ)
    end
    return Czz, Cpm, Cmp
end

function ed_single_particle_benchmark(ψ, sectors, table, L, σ=nothing)
    G = zeros(ComplexF64, L, L)
    A = zeros(ComplexF64, L, L)
    spins = isnothing(σ) ? (1//2, -1//2) : (σ,)
    for spin in spins
        cstates = [matrix(ed_c(i, spin), (sectors.annihilate[spin], sectors.zero), table) * ψ for i in 1:L]
        cdstates = [matrix(ed_cd(i, spin), (sectors.create[spin], sectors.zero), table) * ψ for i in 1:L]
        for i in 1:L, j in 1:L
            G[i,j] += dot(cstates[i], cstates[j])
            A[i,j] += dot(cdstates[i], cdstates[j])
        end
    end
    return G, A
end

function ed_pairing_benchmark(ψ, sectors, table, L, onsite)
    if onsite
        pairs = [(i,) for i in 1:L]
        ops = [ed_onsite_singlet(i) for (i,) in pairs]
    else
        pairs = [(i,j) for i in 1:(L - 1) for j in (i + 1):L]
        ops = [ed_bond_singlet(i,j) for (i,j) in pairs]
    end
    states = [matrix(op, (sectors.minus2, sectors.zero), table) * ψ for op in ops]
    P = [dot(states[a], states[b]) for a in eachindex(pairs), b in eachindex(pairs)]
    return pairs, P
end

function u1_pair_states(gs, L, filling, onsite)
    emu_l = e_min(ComplexF64, U1Irrep, U1Irrep; side=:L, spin=:up, filling=filling)
    emd_l = e_min(ComplexF64, U1Irrep, U1Irrep; side=:L, spin=:down, filling=filling)
    if onsite
        pairs = [(i,) for i in 1:L]
        states = [
            chargedMPS(emd_l, chargedMPS(emu_l, gs, i), i) -
            chargedMPS(emu_l, chargedMPS(emd_l, gs, i), i)
            for (i,) in pairs
        ]
    else
        pairs = [(i,j) for i in 1:(L - 1) for j in (i + 1):L]
        states = [
            chargedMPS(emd_l, chargedMPS(emu_l, gs, j), i) -
            chargedMPS(emu_l, chargedMPS(emd_l, gs, j), i)
            for (i,j) in pairs
        ]
    end
    return pairs, states
end

const BOND_ORDER_PREDICATES = [
    (i,j,k,l) -> i < j < k < l,
    (i,j,k,l) -> i < j == k < l,
    (i,j,k,l) -> i < k < j < l,
    (i,j,k,l) -> i < k < j == l,
    (i,j,k,l) -> i == k < j < l,
    (i,j,k,l) -> i == k < j == l,
    (i,j,k,l) -> i < k < l < j,
    (i,j,k,l) -> i == k < l < j,
    (i,j,k,l) -> k < i < j < l,
    (i,j,k,l) -> k < i < j == l,
    (i,j,k,l) -> k < i < l < j,
    (i,j,k,l) -> k < i == l < j,
    (i,j,k,l) -> k < l < i < j,
]

function random_bond_case(L, predicate)
    while true
        i, j, k, l = rand(1:L, 4)
        if i < j && k < l && predicate(i,j,k,l)
            return (i,j), (k,l)
        end
    end
end

function test_u1_pairing_benchmark(gs, ψed, sectors, table, L, filling)
    errors = Float64[]
    for onsite in (false, true)
        pairs, P = ed_pairing_benchmark(ψed, sectors, table, L, onsite)
        _, states = u1_pair_states(gs, L, filling, onsite)
        if onsite
            P_mps = [dot(states[a], states[b]) for a in eachindex(states), b in eachindex(states)] ./ 2
            @test isapprox(P_mps, P; atol=ED_TOL_OBS)
            push!(errors, maximum(abs.(P_mps - P)))
        else
            bond_errors = Float64[]
            for predicate in BOND_ORDER_PREDICATES
                ij, kl = random_bond_case(L, predicate)
                a, b = findfirst(==(ij), pairs), findfirst(==(kl), pairs)
                value = dot(states[a], states[b]) / 2
                @test isapprox(value, P[a,b]; atol=ED_TOL_OBS)
                push!(bond_errors, abs(value - P[a,b]))
            end
            push!(errors, maximum(bond_errors))
        end
    end
    return errors
end

function test_spin_benchmark(gs, ψed, sectors, table, ops, L)
    Czz, Cpm, Cmp = ed_spin_benchmark(ψed, sectors.zero, table, L)
    Cpm_mps = [correlator(gs, ops.sp_l, ops.sm_r, (i,), (j,)) for i in 1:L, j in 1:L]
    Cmp_mps = [correlator(gs, ops.sm_l, ops.sp_r, (i,), (j,)) for i in 1:L, j in 1:L]
    Cpm_ref = isnothing(ops.sz) ? Czz + (Cpm + Cmp) / 2 : Cpm
    Cmp_ref = isnothing(ops.sz) ? Czz + (Cpm + Cmp) / 2 : Cmp
    @test isapprox(Cpm_mps, Cpm_ref; atol=ED_TOL_OBS)
    @test isapprox(Cmp_mps, Cmp_ref; atol=ED_TOL_OBS)
    if !isnothing(ops.sz)
        Czz_mps = [correlator(gs, ops.sz, ops.sz, (i,), (j,)) for i in 1:L, j in 1:L]
        @test isapprox(Czz_mps, Czz; atol=ED_TOL_OBS)
    end
    return maximum(abs.(Cpm_mps - Cpm_ref)), maximum(abs.(Cmp_mps - Cmp_ref))
end

function test_single_particle_benchmark(gs, ψed, sectors, table, ops, L, σ=nothing)
    G, A = ed_single_particle_benchmark(ψed, sectors, table, L, σ)
    G_mps = [correlator(gs, ops.cp_l, ops.cm_r, (i,), (j,)) for i in 1:L, j in 1:L]
    A_mps = [correlator(gs, ops.cm_l, ops.cp_r, (i,), (j,)) for i in 1:L, j in 1:L]
    @test isapprox(G_mps, G; atol=ED_TOL_OBS)
    @test isapprox(A_mps, -A; atol=ED_TOL_OBS)
    return maximum(abs.(G_mps - G)), maximum(abs.(A_mps + A))
end

function test_pairing_benchmark(gs, ψed, sectors, table, ops, L)
    errors = Float64[]
    for onsite in (false, true)
        pairs, P = ed_pairing_benchmark(ψed, sectors, table, L, onsite)
        sd, sl = onsite ? (ops.sd_onsite, ops.sl_onsite) : (ops.sd, ops.sl)
        if onsite
            P_mps = [correlator(gs, sd, sl, pairs[a], pairs[b]) for a in eachindex(pairs), b in eachindex(pairs)]
            @test isapprox(P_mps, P; atol=ED_TOL_OBS)
            push!(errors, maximum(abs.(P_mps - P)))
        else
            bond_errors = Float64[]
            for predicate in BOND_ORDER_PREDICATES
                ij, kl = random_bond_case(L, predicate)
                a, b = findfirst(==(ij), pairs), findfirst(==(kl), pairs)
                value = correlator(gs, sd, sl, ij, kl)
                @test isapprox(value, P[a,b]; atol=ED_TOL_OBS)
                push!(bond_errors, abs(value - P[a,b]))
            end
            push!(errors, maximum(bond_errors))
        end
    end
    return errors
end

function number_sectors(L, F)
    zero = BinaryBases(2L, ℕ(F))
    annihilate = Dict(1//2 => BinaryBases(2L, ℕ(F - 1)), -1//2 => BinaryBases(2L, ℕ(F - 1)))
    create = Dict(1//2 => BinaryBases(2L, ℕ(F + 1)), -1//2 => BinaryBases(2L, ℕ(F + 1)))
    return (zero=zero, annihilate=annihilate, create=create, minus2=BinaryBases(2L, ℕ(F - 2)))
end

function sz_sectors(L, F)
    sw, su = 1:L, (L + 1):(2L)
    sector(S) = BinaryBases(sw, su, 𝕊ᶻ(S))
    zero = sector(F)
    annihilate = Dict(1//2 => sector(F - 1//2), -1//2 => sector(F + 1//2))
    create = Dict(1//2 => sector(F + 1//2), -1//2 => sector(F - 1//2))
    return (zero=zero, annihilate=annihilate, create=create, minus2=zero)
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
    lattice = Lattice(unitcell, (L,), ('o',))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(lattice))
    table = Table(hilbert, Metric(EDKind(hilbert), hilbert))
    t = Hopping(:t, ComplexF64(-1.0), 1; amplitude = b->b.points[1].site < b.points[2].site ? 1+0.5im : 1-0.5im)
    U = Hubbard(:U, 8.0)
    truncdims = [64, 64, 128, 256, 256, 256]

    @testset "spin-SU2 x particle-U1" begin
        filling = (1, 2)
        F = L ÷ 2
        st = randFiniteMPS(elt, SU2Irrep, U1Irrep, L; filling=filling)
        H = hubbard(elt, SU2Irrep, U1Irrep, Custom(lattice); t=-1-0.5im, U=8, filling=filling)
        gs, env, E0 = dmrg2!(st, H, truncdims; save=false, verbose=false, alg_eigsolve=ED_EIGSOLVE, etol=1e-12)

        ed = ED(lattice, hilbert, (t, U), ℕ(F))
        eigensystem = eigen(ed; nev=1)
        ψed = eigensystem.vectors[1]
        E_ed = eigensystem.values[1]
        sectors = number_sectors(L, F)
        @test isapprox(E0, E_ed; atol=ED_TOL_ENERGY)

        spin_ops = (
            sp_l=S_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling),
            sp_r=S_plus(elt, SU2Irrep, U1Irrep; side=:R, filling=filling),
            sm_l=S_min(elt, SU2Irrep, U1Irrep; side=:L, filling=filling),
            sm_r=S_min(elt, SU2Irrep, U1Irrep; side=:R, filling=filling),
            sz=nothing,
        )
        spin_errors = test_spin_benchmark(gs, ψed, sectors, table, spin_ops, L)

        single_ops = (
            cp_l=e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling),
            cp_r=e_plus(elt, SU2Irrep, U1Irrep; side=:R, filling=filling),
            cm_l=e_min(elt, SU2Irrep, U1Irrep; side=:L, filling=filling),
            cm_r=e_min(elt, SU2Irrep, U1Irrep; side=:R, filling=filling),
        )
        single_errors = test_single_particle_benchmark(gs, ψed, sectors, table, single_ops, L)

        pair_ops = (
            sd=singlet_dagger(elt, SU2Irrep, U1Irrep; filling=filling),
            sl=singlet(elt, SU2Irrep, U1Irrep; filling=filling, side=:R),
            sd_onsite=singlet_dagger(elt, SU2Irrep, U1Irrep; onsite=true, filling=filling),
            sl_onsite=singlet(elt, SU2Irrep, U1Irrep; onsite=true, filling=filling, side=:R),
        )
        pair_errors = test_pairing_benchmark(gs, ψed, sectors, table, pair_ops, L)

        pc = PairCorrelation((pair_ops.sd, pair_ops.sl), Custom(lattice), Neighbors(1=>1), 1, 1)
        wrapped = [correlator(pc, gs; parallel=mode, is=[1], js=[3]) for mode in (1, 2, "np")]
        @test all(x -> eltype(x) === ComplexF64, wrapped)
        @test isapprox(wrapped[1], wrapped[2]; atol=ED_TOL_OBS)
        @test isapprox(wrapped[1], wrapped[3]; atol=ED_TOL_OBS)
        @printf("SU2xU1: |ΔE|=%.2e, spin=%.2e/%.2e, single=%.2e/%.2e, pair=%.2e/%.2e\n",
            abs(E0-E_ed), spin_errors..., single_errors..., pair_errors...)
    end

    @testset "spin-SU2 only, Sz = 1/2 sector" begin
        H = hubbard(elt, SU2Irrep, Custom(lattice); t=-1.0-0.5im, U=8.0)
        vs = repeat([Vect[(FermionParity ⊠ SU2Irrep)]((Int(!isinteger(i)), i) => 1 for i in 0:1//2:1),], L - 1)
        st = FiniteMPS(rand, elt, physicalspace(H), vs; right=Vect[(FermionParity ⊠ SU2Irrep)]((1, 1/2) => 1))
        gs, env, E0 = dmrg2!(st, H, truncdims; save=false, verbose=false, alg_eigsolve=ED_EIGSOLVE, etol=1e-12)

        ed = ED(lattice, hilbert, (t, U), 𝕊ᶻ(1//2))
        eigensystem = eigen(ed; nev=1)
        ψed = eigensystem.vectors[1]
        E_ed = eigensystem.values[1]
        sectors = sz_sectors(L, 1//2)
        @test isapprox(E0, E_ed; atol=ED_TOL_ENERGY)

        spin_ops = (
            sp_l=S_plus(elt, SU2Irrep; side=:L),
            sp_r=S_plus(elt, SU2Irrep; side=:R),
            sm_l=S_min(elt, SU2Irrep; side=:L),
            sm_r=S_min(elt, SU2Irrep; side=:R),
            sz=nothing,
        )
        spin_errors = test_spin_benchmark(gs, ψed, sectors, table, spin_ops, L)

        single_ops = (
            cp_l=e_plus(elt, SU2Irrep; side=:L),
            cp_r=e_plus(elt, SU2Irrep; side=:R),
            cm_l=e_min(elt, SU2Irrep; side=:L),
            cm_r=e_min(elt, SU2Irrep; side=:R),
        )
        single_errors = test_single_particle_benchmark(gs, ψed, sectors, table, single_ops, L)

        pair_ops = (
            sd=singlet_dagger(elt, SU2Irrep),
            sl=singlet(elt, SU2Irrep; side=:R),
            sd_onsite=singlet_dagger(elt, SU2Irrep; onsite=true),
            sl_onsite=singlet(elt, SU2Irrep; onsite=true, side=:R),
        )
        pair_errors = test_pairing_benchmark(gs, ψed, sectors, table, pair_ops, L)
        @printf("SU2:    |ΔE|=%.2e, spin=%.2e/%.2e, single=%.2e/%.2e, pair=%.2e/%.2e\n",
            abs(E0-E_ed), spin_errors..., single_errors..., pair_errors...)
    end

    @testset "spin-U1 x particle-U1" begin
        filling = (1, 1)
        F = L
        st = randFiniteMPS(elt, U1Irrep, U1Irrep, L; filling=filling)
        H = hamiltonian((t, U), lattice, hilbert; filling=filling)
        gs, env, E0 = dmrg2!(st, H, truncdims; save=false, verbose=false, alg_eigsolve=ED_EIGSOLVE, etol=1e-12)

        ed = ED(lattice, hilbert, (t, U), ℕ(F))
        eigensystem = eigen(ed; nev=1)
        ψed = eigensystem.vectors[1]
        E_ed = eigensystem.values[1]
        sectors = number_sectors(L, F)
        @test isapprox(E0, E_ed; atol=ED_TOL_ENERGY)

        spin_ops = (
            sp_l=S_plus(elt, U1Irrep, U1Irrep; side=:L, filling=filling),
            sp_r=S_plus(elt, U1Irrep, U1Irrep; side=:R, filling=filling),
            sm_l=S_min(elt, U1Irrep, U1Irrep; side=:L, filling=filling),
            sm_r=S_min(elt, U1Irrep, U1Irrep; side=:R, filling=filling),
            sz=S_z(elt, U1Irrep, U1Irrep; filling=filling),
        )
        spin_errors = test_spin_benchmark(gs, ψed, sectors, table, spin_ops, L)

        single_errors = zeros(Float64, 2, 2)
        for (row, (σ, spin)) in enumerate(((1//2, :up), (-1//2, :down)))
            single_ops = (
                cp_l=e_plus(elt, U1Irrep, U1Irrep; side=:L, spin=spin, filling=filling),
                cp_r=e_plus(elt, U1Irrep, U1Irrep; side=:R, spin=spin, filling=filling),
                cm_l=e_min(elt, U1Irrep, U1Irrep; side=:L, spin=spin, filling=filling),
                cm_r=e_min(elt, U1Irrep, U1Irrep; side=:R, spin=spin, filling=filling),
            )
            single_errors[row,:] .= test_single_particle_benchmark(gs, ψed, sectors, table, single_ops, L, σ)
        end

        pair_errors = test_u1_pairing_benchmark(gs, ψed, sectors, table, L, filling)
        @printf("U1xU1:  |ΔE|=%.2e, spin=%.2e/%.2e, single=%.2e, pair=%.2e/%.2e\n",
            abs(E0-E_ed), spin_errors..., maximum(single_errors), pair_errors...)
    end
    
    @testset "triangular-lattice form factors and 120-degree spin" begin
        filling = (1, 1)
        Ntri = 8
        tri_unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.5, sqrt(3)/2]])
        tri_lattice = Lattice(tri_unitcell, (2, 4), ('p', 'p'))
        tri_hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(tri_lattice))
        tri_table = Table(tri_hilbert, Metric(EDKind(tri_hilbert), tri_hilbert))
        H = hubbard(elt, SU2Irrep, U1Irrep, Custom(tri_lattice);
            t=-1-0.5im, U=8, filling=filling)
        st = randFiniteMPS(elt, SU2Irrep, U1Irrep, Ntri; filling=filling)
        gs, env, E0 = dmrg2!(st, H, truncdims;
            save=false, verbose=false, alg_eigsolve=ED_EIGSOLVE, etol=1e-12)

        ed = ED(tri_lattice, tri_hilbert, (t, U), ℕ(Ntri))
        eigensystem = eigen(ed; nev=1)
        ψed = eigensystem.vectors[1]
        E_ed = eigensystem.values[1]
        sectors = number_sectors(Ntri, Ntri)
        @test isapprox(E0, E_ed; atol=ED_TOL_ENERGY)

        sld = singlet_dagger(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
        sl = singlet(elt, SU2Irrep, U1Irrep; side=:R, filling=filling)
        tri_bonds = bonds(tri_lattice, Neighbors(1=>1))
        center_bonds = [filter(b -> any(p -> p.site == i, b.points), tri_bonds) for i in 1:Ntri]
        channels = (
            extended_s = b -> 1.0,
            d1 = b -> cos(2 * azimuth(rcoordinate(b))),
            d2 = b -> sin(2 * azimuth(rcoordinate(b))),
            chiral = b -> cis(2 * azimuth(rcoordinate(b))),
        )
        bond_indices(b) = Tuple(sort([p.site for p in b.points]))
        for f in channels, b in tri_bonds
            @test isapprox(f(reverse(b)), f(b); atol=1e-12)
        end

        center_pairs = [(1, 1), (1, 2), (1, 8), (2, 7), (4, 5)]
        for name in propertynames(channels)
            f = getproperty(channels, name)
            states = map(1:Ntri) do r
                op = sum(f(b) * ed_bond_singlet(bond_indices(b)...) for b in center_bonds[r])
                return matrix(op, (sectors.minus2, sectors.zero), tri_table) * ψed
            end
            for (r, s) in center_pairs
                expected = dot(states[r], states[s])
                pc = PairCorrelation((sld, sl), Custom(tri_lattice), Neighbors(1=>1), 1, 1;
                    amplitude=f, intralayer=nothing)
                value = correlator(pc, gs; parallel=1, is=[r], js=[s])[1]
                @test isapprox(value, expected; atol=ED_TOL_OBS)
            end
        end

        pc = PairCorrelation((sld, sl), Custom(tri_lattice), Neighbors(1=>1), 1, 1;
            amplitude=channels.chiral, intralayer=nothing)
        serial = correlator(pc, gs; parallel=1, is=[1], js=[8])
        threaded = correlator(pc, gs; parallel=2, is=[1], js=[8])
        @test isapprox(serial, threaded; atol=ED_TOL_OBS)

        spin_ops = (
            sp_l=S_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling),
            sp_r=S_plus(elt, SU2Irrep, U1Irrep; side=:R, filling=filling),
            sm_l=S_min(elt, SU2Irrep, U1Irrep; side=:L, filling=filling),
            sm_r=S_min(elt, SU2Irrep, U1Irrep; side=:R, filling=filling),
            sz=nothing,
        )
        test_spin_benchmark(gs, ψed, sectors, tri_table, spin_ops, Ntri)
        Czz, Cpm, Cmp = ed_spin_benchmark(ψed, sectors.zero, tri_table, Ntri)
        C_ed = Czz + (Cpm + Cmp) / 2
        C_mps = [correlator(gs, spin_ops.sp_l, spin_ops.sm_r, (i,), (j,))
            for i in 1:Ntri, j in 1:Ntri]
        rs = [tri_lattice[i] for i in 1:Ntri]
        ks = [[4π/3, 0.0], [-4π/3, 0.0]]
        sf_ed = static_structure_factor(C_ed, rs, ks)
        sf_mps = static_structure_factor(C_mps, rs, ks)
        @test isapprox(sf_mps, sf_ed; atol=ED_TOL_OBS)
        @printf("triangular: |ΔE|=%.2e, S(K) error=%.2e\n", abs(E0-E_ed),
            maximum(abs.(sf_mps-sf_ed)))
    end

    @testset "fast chargedMPS compression" begin
        filling = (1, 2)
        st = randFiniteMPS(elt, SU2Irrep, U1Irrep, N; filling=filling)
        H = hubbard(elt, SU2Irrep, U1Irrep, Custom(lattice); t=-1-0.5im, U=8, filling=filling)
        gs, env, E0 = dmrg2!(st, H, [64, 64, 128]; save=false, verbose=false)

        cm = e_min(elt, SU2Irrep, U1Irrep; filling=filling)

        # two-site compression (RAM and disk-backed environments) must reproduce
        # the exact chargedMPO contraction to machine precision: the initial
        # guess inherits gs's bond spaces and truncrank(128) keeps everything
        for j in (3, 6)
            ket_exact = chargedMPS(cm, gs, j)
            ket_ram   = chargedMPS(cm, gs, j, myDMRG2(; trunc=truncrank(128), maxiter=20))
            ket_disk  = chargedMPS(cm, gs, j, myDMRG2(; trunc=truncrank(128), maxiter=20); disk=true)
            @test abs(dot(ket_exact, ket_ram)) ≈ norm(ket_exact) * norm(ket_ram) atol=1e-10
            @test abs(dot(ket_exact, ket_disk)) ≈ norm(ket_exact) * norm(ket_disk) atol=1e-10
            @test sweep_dot(gs, cm, ket_ram) ≈ sweep_dot(gs, cm, ket_exact) atol=1e-10
        end

        # in-place form with an explicit mixed HalfFiniteEnvironments
        ψ₀ = randFiniteMPS(eltype(gs[1]), gs, cm)
        envs0 = HalfFiniteEnvironments(ψ₀, chargedMPO(cm, 5, N), gs)
        ψ_ip, envs0, ϵ = chargedMPS!(ψ₀, cm, gs, 5, myDMRG2(; trunc=truncrank(128), maxiter=20))
        ket_exact = chargedMPS(cm, gs, 5)
        @test abs(dot(ket_exact, ψ_ip)) ≈ norm(ket_exact) * norm(ψ_ip) atol=1e-10
        @test ϵ < 1e-6

        # one-site compression cannot grow the bond dimension (lossy by
        # construction, but must stay close at site 6): a `DMRG` polish has a
        # non-truncating gauge, so the zip-up warm start needs an explicit
        # truncation — the bonds are then capped by `alg_zipup`, not by `gs`
        ket1 = chargedMPS(cm, gs, 6, myDMRG1(; maxiter=30); alg_zipup=Zipup(; trunc=truncrank(128)))
        ket_exact = chargedMPS(cm, gs, 6)
        @test all(i -> dim(left_virtualspace(ket1, i)) <= 128, 1:N)
        @test abs(dot(ket_exact, ket1)) / (norm(ket_exact) * norm(ket1)) > 0.8
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

    @testset "fast TDVP integration diagnostics" begin
        N = 4
        H = hubbard(ComplexF64, SU2Irrep, U1Irrep, FiniteChain(N);
                    t = -1.0, U = 8.0, filling = (1, 1))
        st = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, N; filling = (1, 1))
        for alg in (myTDVP1(; krylovdim = 12),
                    myTDVP2(; trunc = truncrank(64), krylovdim = 12))
            ψref = complex(copy(st))
            ψfast = complex(copy(st))
            envref = MPSKit.environments(ψref, H, ψref)
            envfast = HalfFiniteEnvironments(ψfast, H)
            ψref, envref, _ = MPSKit.timestep!(ψref, H, 0.0, 0.02, alg, envref)
            infos = Any[]
            ψfast, envfast = fast_timestep!(ψfast, H, 0.0, 0.02, alg, envfast;
                                            integration_stats = infos)
            @test abs(dot(ψfast, ψref)) ≈ norm(ψfast) * norm(ψref) atol = 1e-9
            @test length(infos) == (alg isa TDVP ? 4N - 2 : 4N - 6)
            @test count(info -> info.kind == :AC2, infos) ==
                  (alg isa TDVP ? 0 : 2N - 2)
            @test count(info -> info.kind == :C, infos) ==
                  (alg isa TDVP ? 2N - 2 : 0)
            @test all(info -> info.converged == 1, infos)
            @test all(info -> info.krylovdim >= 0, infos)
            @test all(info -> info.numops >= 1 && 0 <= info.t_mv <= info.t_total, infos)
        end
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

@testset "DMRG update-level checkpoint and resume" begin
    Ly, Lx = 2, 4
    filling = (1, 1)
    coords = snake_2D([[0.0, 1.0], [sqrt(3) / 2, 1 / 2]],
        [o for j in 0:(Lx - 1) for o in (j < Lx - 1 ?
            [fill(iseven(j) ? 1 : -1, Ly - 1); 2] :
            fill(iseven(j) ? 1 : -1, Ly - 1))])
    lattice = Lattice(coords...; vectors = [[0.0, Float64(Ly)]])
    N = length(lattice)
    hilbert = Hilbert(site => Fock{:f}(1, 2) for site = 1:N)
    H = hamiltonian((Hopping(:t, -1.0, 1), Hubbard(:U, 8.0)), lattice, hilbert; filling)
    ψ0 = complex(randFiniteMPS(ComplexF64, U1Irrep, U1Irrep, N; filling))
    Ds = [16, 32]

    function run_captured(f)
        oldout = stdout
        rd, wr = redirect_stdout()
        local out
        try
            f()
            redirect_stdout(oldout)
            close(wr)
            out = read(rd, String)
        finally
            redirect_stdout(oldout)
        end
        return out
    end

    mktempdir() do root
        # uninterrupted reference (in-RAM environments)
        ψref = copy(ψ0)
        _, _, Eref = dmrg2!(ψref, H, Ds; filename = joinpath(root, "ref.jld2"),
            etol = 1e-12, max_tailsweeps = 2, verbose = 0)

        # (a) kill mid-L2R of sweep 1, then resume from a fresh copy of the
        # initial state with disk-backed environments (a process rerun)
        dir1 = joinpath(root, "a")
        ckpt1 = joinpath(dir1, "run.jld2")
        mkpath(dir1)
        ψi = copy(ψ0)
        @test_throws InterruptException dmrg2!(ψi, H, Ds;
            filename = ckpt1, disk = joinpath(dir1, "envs"),
            etol = 1e-12, max_tailsweeps = 2, verbose = 0, _interrupt_after = 5)
        @test isdir(joinpath(dir1, "envs", "run.jld2.updates.d"))
        ψr = copy(ψ0)
        out = run_captured() do
            dmrg2!(ψr, H, Ds; filename = ckpt1, disk = joinpath(dir1, "envs"),
                etol = 1e-12, max_tailsweeps = 2, verbose = 1)
        end
        @test occursin("resuming sweep 1 from right position 5", out)
        @test occursin(r"environments reused: [1-9]", out)
        @test abs(dot(ψr, ψref)) ≈ norm(ψr) * norm(ψref) atol = 1e-8
        @test !isdir(joinpath(dir1, "envs", "run.jld2.updates.d"))

        # (b) kill exactly at the sweep boundary (sweep 1 complete), resume
        # continues at sweep 2
        dir2 = joinpath(root, "b")
        ckpt2 = joinpath(dir2, "run.jld2")
        mkpath(dir2)
        ψb = copy(ψ0)
        @test_throws InterruptException dmrg2!(ψb, H, Ds;
            filename = ckpt2, disk = joinpath(dir2, "envs"),
            etol = 1e-12, max_tailsweeps = 2, verbose = 0, _interrupt_after = 13)
        ψb2 = copy(ψ0)
        out = run_captured() do
            dmrg2!(ψb2, H, Ds; filename = ckpt2, disk = joinpath(dir2, "envs"),
                etol = 1e-12, max_tailsweeps = 2, verbose = 1)
        end
        @test occursin("resuming sweep 1 (sweep complete)", out)
        @test abs(dot(ψb2, ψref)) ≈ norm(ψb2) * norm(ψref) atol = 1e-8

        # (c) DMRG1 + CBE (deterministic rsvd=false), killed in the R2L half
        alg_expand = CBEExpand(; trunc = truncrank(2), rsvd = false)
        ψref1 = copy(ψ0)
        _, _, Eref1 = dmrg1!(ψref1, H, Ds; alg_expand,
            filename = joinpath(root, "ref1.jld2"),
            etol = 1e-12, max_tailsweeps = 2, verbose = 0)
        dir3 = joinpath(root, "c")
        ckpt3 = joinpath(dir3, "run.jld2")
        mkpath(dir3)
        ψc = copy(ψ0)
        @test_throws InterruptException dmrg1!(ψc, H, Ds; alg_expand,
            filename = ckpt3, disk = joinpath(dir3, "envs"),
            etol = 1e-12, max_tailsweeps = 2, verbose = 0, _interrupt_after = 10)
        ψc2 = copy(ψ0)
        _, _, Ec = dmrg1!(ψc2, H, Ds; alg_expand,
            filename = ckpt3, disk = joinpath(dir3, "envs"),
            etol = 1e-12, max_tailsweeps = 2, verbose = 0)
        @test Ec ≈ Eref1 atol = 1e-8
        @test abs(dot(ψc2, ψref1)) ≈ norm(ψc2) * norm(ψref1) atol = 1e-8

        # (d) a mismatching schedule discards the checkpoint directory and
        # starts fresh
        dir4 = joinpath(root, "d")
        ckpt4 = joinpath(dir4, "run.jld2")
        mkpath(dir4)
        ψd = copy(ψ0)
        @test_throws InterruptException dmrg2!(ψd, H, Ds;
            filename = ckpt4, etol = 1e-12, max_tailsweeps = 2, verbose = 0,
            _interrupt_after = 3)
        ψd2 = copy(ψ0)
        @test_logs (:info, r"update-level checkpoint discarded") match_mode = :any begin
            dmrg2!(ψd2, H, [16, 64]; filename = ckpt4,
                etol = 1e-12, max_tailsweeps = 2, verbose = 0)
        end
        @test !isdir(ckpt4 * ".updates.d")
    end
end
