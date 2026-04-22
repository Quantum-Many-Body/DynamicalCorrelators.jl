# Benchmark: FiniteMPS CBE-DMRG1 on 4×6 Hubbard model with U₁×U₁ symmetry
# Same model as src/draft/benchmark_cbe_vs_dmrg2.jl for comparison

using LinearAlgebra
BLAS.set_num_threads(1)

using FiniteMPS
using FiniteLattices
using TensorKit
using TensorKit:×
# ── Model parameters (same as DynamicalCorrelators benchmark) ──
Lx = 6        # length
Ly = 4        # width (YC periodic direction)
N  = Lx * Ly  # 24 sites
t_hop = 1.0
U_int = 8.0

println("="^60)
println("  Hubbard model: $(Ly)×$(Lx) square lattice (YCSqua)")
println("  N=$N, t=$t_hop, U=$U_int, half-filling, U₁×U₁ symmetry")
println("="^60)

# ── Lattice ──
Latt = OpenSqua(Lx, Ly) #|> Snake!
L = size(Latt)

# ── Hamiltonian via InteractionTree ──
Tree = InteractionTree(L)

# Hopping: -t ∑_<i,j>,σ (c†_iσ c_jσ + h.c.)
for (i, j) in neighbor(Latt; ordered=true)
    addIntr!(Tree, U1U1Fermion.FdagF₊, (i, j), (true, true), -t_hop;
             Z=U1U1Fermion.Z, name=(:Fdag₊, :F₊))
    addIntr!(Tree, U1U1Fermion.FdagF₋, (i, j), (true, true), -t_hop;
             Z=U1U1Fermion.Z, name=(:Fdag₋, :F₋))
end

# On-site: U ∑_i n_i↑ n_i↓
for i in 1:L
    addIntr!(Tree, U1U1Fermion.nd, i, U_int; name=:nd)
end

H = AutomataMPO(Tree)

# ── Initial MPS (small bond dim) ──
aspace = Rep[U₁×U₁]((c, s) => 1 for c in -1:1:1 for s in -1:1//2:1)
Ψ = randMPS(Float64, L, U1U1Fermion.pspace, aspace)
Env = Environment(Ψ', H, Ψ)

# ── Truncation schedule (same as benchmark) ──
lsD = [64, 64, 256, 256, 1024, 1024]


# ── CBE-DMRG1 sweeps ──
println("\n── FiniteMPS.jl CBE-DMRG1 ──")
t_start = time()
for (iter, D) in enumerate(lsD)
    t_sweep = time()
    info, timer = DMRGSweep1!(Env;
        CBEAlg=FullCBE(),
        #CBEAlg=NaiveCBE(D + div(D, 2), 1e-8; rsvd=true),
        trunc=truncdim(D),
        noise=(0.5, 0.0),
        K=16,
        verbose=0,
        GCsweep=true)

    Eg = info[2].dmrg[1].Eg
    dt = time() - t_sweep
    println("[$iter/$(length(lsD))] D=$D, E₀=$Eg, time=$(round(dt; digits=2))s")
    flush(stdout)
end
t_total = time() - t_start
println("\nTotal time: $(round(t_total; digits=2))s")


# Julia Version 1.12.5
# FiniteMPS Version 1.8.0
# Multi-threading Info:
#  Julia: 8
#  BLAS: 1
#  action: 8
#  mul: 1
#  svd: 8
#  eig: 8
# BLAS Info:
#  LBTConfig([ILP64] libopenblas64_.dylib)
# ============================================================
#   Hubbard model: 4×6 square lattice (YCSqua)
#   N=24, t=1.0, U=8.0, half-filling, U₁×U₁ symmetry
# ============================================================

# ── FiniteMPS.jl CBE-DMRG1 ──
# [1/6] D=64, E₀=-9.114597145457505, time=28.49s
# [2/6] D=64, E₀=-10.099471418660887, time=3.31s
# [3/6] D=256, E₀=-10.518607882664336, time=4.65s
# [4/6] D=256, E₀=-10.52482314693454, time=5.79s
# [5/6] D=1024, E₀=-10.596158504571866, time=67.96s
# [6/6] D=1024, E₀=-10.59621966983013, time=128.11s

# Total time: 238.34s