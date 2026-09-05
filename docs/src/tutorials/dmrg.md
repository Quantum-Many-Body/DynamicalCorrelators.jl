# Ground State with DMRG

DynamicalCorrelators.jl provides two finite-system DMRG drivers, `dmrg2` and
`dmrg1`, built on MPSKit's per-site update machinery. Both take an explicit
`truncdims` schedule that fixes the number of sweeps and the target bond
dimension of each sweep, print per-move and per-sweep diagnostics, and write
JLD2 checkpoints along the way.

Use this page to choose the ground-state workflow before computing dynamical
correlators.

## Standard Two-Site DMRG

`dmrg2` runs two-site DMRG with one sweep (a left-to-right plus right-to-left
pass) per entry of `truncdims`:

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

gs, envs, E0 = dmrg2(ψ0, H, truncdims;
    filename = "dmrg2.jld2",
    verbose = true,
)
```

`dmrg2!` is the mutating version. Each saved sweep stores the state, energy,
per-bond Galerkin and truncation errors, and the maximal bond dimension in the
JLD2 file, which is useful for long runs on a cluster.

## One-Site DMRG with CBE

One-site DMRG is cheaper than two-site DMRG, but it cannot grow the bond space
by itself. `dmrg1` therefore runs MPSKit's one-site `DMRG` update with a bond
expansion ahead of each eigensolve. With the default `alg_expand =
OptimalExpand` setting, each bond is enlarged ahead of the one-site eigensolve
by up to `ceil(Int, delta*D)` directions selected from the projected two-site
update, and truncated back to `D = truncdims[iter]` by the gauge step:

```julia
gs, envs, E0 = dmrg1(ψ0, H, truncdims;
    delta = 0.1,
    filename = "dmrg1.jld2",
    verbose = 2,
)
```

Any bond-expansion algorithm defined by MPSKit can be plugged in by passing an
instance, e.g. `alg_expand = SketchedExpand(; trunc = truncrank(64))` for the
randomized single-site-cost selection, or `alg_expand = RandExpand(...)`. Pass
`alg_expand = nothing` for plain single-site DMRG (which cannot grow the bond).

## Logging and Checkpoints

- `verbose = 1` prints one summary line per sweep (energy, maximal bond
  dimension, energy change, maximal Galerkin and truncation errors); a
  `TimerOutput` summary is printed at the end.
- `verbose = 2` additionally prints one line per local update (sweep direction,
  bond, current bond dimension, local errors, timestamp).
- `save` controls JLD2 checkpointing (`false` writes nothing, `true`
  stores only the final sweep, a vector of sweep indices stores those
  sweeps), together with `filename`.

## Hybrid Two-Site + One-Site (CBE) DMRG

Two-site updates grow the bond by up to a factor of the physical dimension per
update, so they deliver fast, robust bond growth at small D where sweeps are
cheap; one-site sweeps with bond expansion are much cheaper at large D.
`dmrg_mix` combines both in a single run with continuous sweep numbering and
checkpointing:

```julia
ψ, envs, E0 = dmrg_mix(ψ0, H, [64, 128, 256], [512, 1024, 1024];
    delta = 0.1, filename = "dmrg_mix.jld2")
```

or with a single schedule and a switch point (entries `≤ switch_D` run
two-site, the rest one-site with CBE):

```julia
ψ, envs, E0 = dmrg_mix(ψ0, H, [64, 128, 256, 512, 1024, 1024]; switch_D = 256)
```

## Custom Algorithm Choices

By default the local eigensolver is adaptive (MPSKit's `AdaptiveKrylov`:
tolerance, Krylov dimension and restart count are retuned per local update).
To pin it down, pass an explicit solver:

```julia
alg_eigsolve = Lanczos(; krylovdim = 24, maxiter = 1, tol = 1e-8,
    orth = ModifiedGramSchmidt(), eager = true, verbosity = 0)
gs, envs, E0 = dmrg1(ψ0, H, truncdims; alg_eigsolve, alg_svd = LAPACK_DivideAndConquer())
```

The `my*` constructors remain available when you want a preconfigured MPSKit
algorithm object for use with `find_groundstate` or `time_evolve` directly:

```julia
alg_dmrg1 = myDMRG1(; tol = 1e-8, maxiter = 100)                    # adaptive eigensolver
alg_dmrg2 = myDMRG2(; tol = 1e-6, maxiter = 50, trunc = truncrank(1024))
alg_cbe = myDMRG1_CBE(; tol = 1e-6, maxiter = 100, D = 1024, delta = 0.1)
alg_fixed = myDMRG1(; tol = 1e-8, maxiter = 100, adaptive = false, krylovdim = 16)
```

## When to Use Which Method

- Use `dmrg2` for small and medium calculations, debugging, or when you want a
  conventional robust two-site warmup.
- Use `dmrg1` for larger finite systems where the one-site cost matters and
  bond growth must remain controlled.
- Use `dmrg_mix` to get both in one run: two-site growth at small D, one-site
  CBE sweeps at large D.
- Use `find_groundstate` from MPSKit directly when you need convergence-based
  stopping (`tol`/`maxiter`) instead of a fixed sweep schedule.

## Infinite DMRG

For translation-invariant systems, use MPSKit's `IDMRG`/`IDMRG2` directly:

```julia
using MPSKitModels: InfiniteChain

H∞ = hubbard(Float64, SU2Irrep, U1Irrep, InfiniteChain(2);
    t = 1.0, U = 8.0, filling = filling)

ψ∞ = randInfiniteMPS(ComplexF64, SU2Irrep, U1Irrep, 2; filling)
gs∞, envs∞, ϵ∞ = find_groundstate(ψ∞, H∞, IDMRG2(; trunc = truncrank(512)))
```

## Practical Notes

- Keep BLAS threads modest when also using Julia threads or distributed workers.
  The package sets BLAS threads to one at initialization.
- Checkpoint files are ordinary JLD2 files. You can inspect sweep energies,
  truncation errors, and saved states without rerunning the calculation.
- The CBE parameters `D` and `delta` should be chosen together: `D` is the
  target kept dimension, while `delta` controls the temporary working space.
- MPSKit's sweeps allocate their local-update scratch space from a dedicated
  allocator by default; see `MPSKit.Defaults.set_buffering!` and
  `MPSKit.Defaults.set_scheduler!` for memory/threading controls.
