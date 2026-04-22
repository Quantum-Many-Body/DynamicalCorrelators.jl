using Revise
using Pkg
Pkg.activate(".")
using LinearAlgebra
BLAS.set_num_threads(1)
"""
Benchmark: CBE + 1-site DMRG vs 2-site DMRG on a 4×6 square lattice Hubbard model.
"""

using DynamicalCorrelators
using TensorKit
using MPSKit
using MPSKitModels: FiniteStrip
using Printf

# ── Model: half-filled Hubbard on 4×6 square lattice ──
Lx = 6         # length
Ly = 4         # width
N  = Lx * Ly   # total sites = 24
t_hop = 1.0
U_int = 8.0

println("="^60)
println("  Hubbard model: $(Ly)×$(Lx) square lattice (FiniteStrip)")
println("  N=$N, t=$t_hop, U=$U_int, half-filling")
println("="^60)

# Build Hamiltonian on a finite strip (open boundaries)
lattice = FiniteStrip(Ly, N)
H = hubbard(Float64, U1Irrep, U1Irrep, lattice; t=t_hop, U=U_int, μ=0.0, filling=(1,1))

# Build random initial MPS (small bond dim)
ψ_init = randFiniteMPS(Float64, U1Irrep, U1Irrep, N; filling=(1,1))

# ── Truncation schedule ──
truncdims = [64, 64, 256, 256, 1024, 1024]

println("\n── CBE + 1-site DMRG ──")
ψ1 = copy(ψ_init)

ψ1, envs1, E1_final = dmrg1_cbe!(ψ1, H, truncdims;delta=0.1,
    filename="draft_dmrg1_cbe_test.jld2", verbose=1);
println("finished")

# ============================================================
#   Hubbard model: 4×6 square lattice (FiniteStrip)
#   N=24, t=1.0, U=8.0, half-filling
# ============================================================

# ── CBE + 1-site DMRG ──
# CBE-DMRG1 Sweep Started: 22.Apr 2026 16:38
# [1/6] CBE-DMRG1 sweep | duration: 3 seconds, 525 milliseconds
#   E₀ = -9.5699390037 | D =   64 | ΔE = 5.885e+01 | ϵp = 1.625e+00 | err² = 2.685e-04
# [2/6] CBE-DMRG1 sweep | duration: 4 seconds, 732 milliseconds
#   E₀ = -10.0178418500 | D =   64 | ΔE = 4.479e-01 | ϵp = 1.177e+00 | err² = 1.650e-04
# [3/6] CBE-DMRG1 sweep | duration: 8 seconds, 145 milliseconds
#   E₀ = -10.4506952247 | D =  256 | ΔE = 4.329e-01 | ϵp = 8.046e-01 | err² = 6.199e-06
# [4/6] CBE-DMRG1 sweep | duration: 9 seconds, 706 milliseconds
#   E₀ = -10.5176702391 | D =  256 | ΔE = 6.698e-02 | ϵp = 4.142e-01 | err² = 2.582e-05
# [5/6] CBE-DMRG1 sweep | duration: 58 seconds, 453 milliseconds
#   E₀ = -10.5914846626 | D = 1024 | ΔE = 7.381e-02 | ϵp = 4.287e-01 | err² = 9.354e-07
# [6/6] CBE-DMRG1 sweep | duration: 1 minute, 27 seconds, 582 milliseconds
#   E₀ = -10.5962092176 | D = 1024 | ΔE = 4.725e-03 | ϵp = 1.792e-01 | err² = 2.316e-06
# Ended: 22.Apr 2026 16:40 | total duration: 2 minutes, 52 seconds, 554 milliseconds
# ───────────────────────────────────────────────────────────────────────
#                               Time                    Allocations      
#                      ───────────────────────   ────────────────────────
#   Tot / % measured:        173s /  97.6%            219GiB /  99.7%    

# Section      ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────
# eigsolve        276    98.2s   58.2%   356ms    148GiB   67.7%   549MiB
# CBE expand      276    69.0s   40.8%   250ms   69.9GiB   32.0%   259MiB
# SVD trunc       276    1.70s    1.0%  6.15ms    837MiB    0.4%  3.03MiB
# ───────────────────────────────────────────────────────────────────────
