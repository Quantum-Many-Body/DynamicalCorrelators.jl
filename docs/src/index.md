# DynamicalCorrelators.jl

DynamicalCorrelators.jl is a frontend for matrix-product-state calculations of
ground states, real-time dynamical correlation functions, finite-temperature
correlators, and momentum-frequency spectral functions.

The package is built around three layers:

- symbolic lattice and model construction through
  [QuantumLattices.jl](https://github.com/Quantum-Many-Body/QuantumLattices.jl)
  and model helpers in this package;
- finite and infinite MPS algorithms from
  [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl);
- convenience workflows for charged states, TDVP time evolution, checkpointed
  dynamical correlators, and Fourier transforms.

## Core Features

DynamicalCorrelators.jl provides convenience wrappers for the common pieces of
finite-MPS dynamical-correlation workflows:

- `myDMRG1_CBE` configures one-site DMRG with Controlled Bond Expansion (CBE)
  through MPSKit's `DMRG(; alg_expand = OptimalExpand(...))`.
- `myTDVP1_CBE` configures CBE-assisted one-site TDVP through MPSKit's
  `TDVP(; alg_expand = OptimalExpand(...), trunc = ...)`. This lets single-site
  TDVP grow bonds through CBE while keeping the cheaper one-site time-evolution
  sweep.
- `dcorrelator` supports single-source and multi-source checkpointed
  real-time correlator calculations.
- finite-temperature correlators read a saved `rho(t)` trajectory one slice at
  a time and use sweep contractions against the active charged ket.

## Basic Workflow

A typical zero-temperature calculation is:

1. Build an MPO Hamiltonian.
2. Find a ground state with DMRG or CBE-DMRG1.
3. Build local operators such as `e_plus`, `e_min`, or `S_plus`.
4. Compute real-space/time correlators with `dcorrelator`.
5. Transform to spectra with `fourier_kw` or `fourier_rw`.

```julia
using TensorKit
using MPSKit
using MPSKitModels: FiniteChain
using DynamicalCorrelators

N = 24
filling = (1, 1)
H = hubbard(Float64, SU2Irrep, U1Irrep, FiniteChain(N);
    t = 1.0, U = 8.0, filling = filling)

ψ0 = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, N; filling)
gs, envs, E = dmrg2(ψ0, H, [128, 256, 512])

times = 0:0.05:10
sp = S_plus(Float64, SU2Irrep, U1Irrep; filling)

gf = dcorrelator(gs, H, sp, 1:N;
    times,
    tdvp1 = myTDVP1_CBE(D = 512),
    tdvp2 = myTDVP1_CBE(D = 512),
)
```

For one source channel, pass an integer `id`:

```julia
gf_site = dcorrelator(gs, H, sp, div(N, 2);
    times,
    record_indices = 1:101,
    tdvp1 = myTDVP1_CBE(D = 512),
)
```

## Guide

- [Getting Started](tutorials/getting_started.md): the package layout and a
  compact end-to-end workflow.
- [Ground State with DMRG](tutorials/dmrg.md): `dmrg1`/`dmrg2` with explicit
  `truncdims` schedules and the default algorithm constructors.
- [Dynamical Correlations](tutorials/dynamical_correlations.md):
  zero-temperature real-time correlators and CBE-TDVP1.
- [Spectral Functions](tutorials/spectral_functions.md): real-space/time to
  momentum-frequency transforms.
- [Finite Temperature](tutorials/finite_temperature.md): purification,
  imaginary-time preparation, and finite-temperature correlators.

## API Reference

The API pages list exported models, operators, states, algorithms, observables,
and utility functions. Start with:

- [Algorithms](api/algorithms.md)
- [States](api/states.md)
- [Observables](api/observables.md)
- [Operators](api/operators.md)

## Compatibility Note

Before v1.0, minor versions may change APIs when the internal workflow improves.
Use the exported `my*` constructors (`myDMRG2`, `myDMRG1_CBE`, `myTDVP`,
`myTDVP1_CBE`, `myTDVP2`) for the package's default algorithm configurations.

## Acknowledgments

This package builds on MPSKit.jl, TensorKit.jl, MPSKitModels.jl, and
QuantumLattices.jl. We thank the developers of those packages and the users who
have helped test the CBE and dynamical-correlation workflows.
