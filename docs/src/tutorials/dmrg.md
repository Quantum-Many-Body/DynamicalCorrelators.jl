# Ground State with DMRG

DynamicalCorrelators.jl uses MPSKit.jl for the underlying DMRG algorithms and
adds wrappers for checkpointed finite-system sweeps and Controlled Bond
Expansion (CBE).

Use this page to choose the ground-state workflow before computing dynamical
correlators.

## Standard Two-Site DMRG

The `dmrg2` wrapper runs a sequence of two-site DMRG sweeps with increasing
target bond dimensions:

```julia
using TensorKit
using MPSKit
using MPSKitModels: FiniteChain
using DynamicalCorrelators

N = 32
filling = (1, 1)

H = hubbard(Float64, SU2Irrep, U1Irrep, FiniteChain(N);
    t = 1.0, U = 8.0, filling = filling)

ψ0 = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, N; filling)
truncdims = [128, 256, 512, 1024]

gs, envs, ϵ = dmrg2(ψ0, H, truncdims;
    alg = myDMRG2(),
    filename = "dmrg2.jld2",
    verbose = true,
)
```

`dmrg2!` is the mutating version. It stores sweep states and diagnostics in the
JLD2 file, which is useful for long runs on a cluster.

## One-Site DMRG with CBE

One-site DMRG is cheaper than two-site DMRG, but it cannot grow the bond space by
itself. Controlled Bond Expansion fixes that by enlarging the active bond before
the one-site eigensolve and truncating back to the target dimension afterwards.

```julia
gs, envs, ϵ = dmrg1_cbe(ψ0, H, truncdims;
    cbe_method = :direct,
    delta = 0.1,
    cbe_tol = 1e-10,
    filename = "dmrg1_cbe.jld2",
    verbose = 2,
)
```

The default direct projector is the production path used in v0.11. For finite
Jordan-MPO Hamiltonians it dispatches to a sparse-channel multithreaded
projector, so the expensive `AA`-channel contractions can be distributed across
Julia threads.

Use `project_error = true` when you want explicit CBE projection-error
diagnostics. It is off by default because it adds extra two-site contraction
work.

## Algorithm Constructors

The package exports small constructors for common algorithm choices:

```julia
alg_dmrg1 = myDMRG(; tol = 1e-8, maxiter = 100, krylovdim = 16)
alg_dmrg2 = myDMRG2(; tol = 1e-6, maxiter = 50, trunc = truncrank(1024))
```

For CBE-DMRG1, the local eigensolver default is:

```julia
myDMRG1CBE_eigsolve
```

Pass it through `alg_eigsolve` if you want to customize the Krylov parameters.

## When to Use Which Method

- Use `dmrg2` for small and medium calculations, debugging, or when you want a
  conventional robust two-site warmup.
- Use `dmrg1_cbe` for larger finite systems where the one-site cost matters and
  bond growth must remain controlled.
- Use `find_groundstate` from MPSKit directly when you need an algorithm path not
  wrapped here.

## Infinite DMRG

For translation-invariant systems, `idmrg2` wraps MPSKit's infinite-system
algorithms with similar logging and checkpointing:

```julia
using MPSKitModels: InfiniteChain

H∞ = hubbard(Float64, SU2Irrep, U1Irrep, InfiniteChain(2);
    t = 1.0, U = 8.0, filling = filling)

ψ∞ = randInfiniteMPS(ComplexF64, SU2Irrep, U1Irrep, 2; filling)
gs∞, envs∞, ϵ∞ = idmrg2(ψ∞, H∞; filename = "idmrg2.jld2")
```

## Practical Notes

- Keep BLAS threads modest when also using Julia threads or distributed workers.
  The package sets BLAS threads to one at initialization.
- Checkpoint files are ordinary JLD2 files. You can inspect sweep energies,
  truncation errors, and saved states without rerunning the calculation.
- CBE parameters `D`, `delta`, and `cbe_tol` should be chosen together: `D` is
  the target kept dimension, while `delta` controls the temporary working space.
