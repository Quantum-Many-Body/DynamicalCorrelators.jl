# Getting Started

This tutorial gives the shortest route from an MPO Hamiltonian to a dynamical
correlation function. The examples use a finite Hubbard chain with
`SU2Irrep × U1Irrep` symmetry, but the same workflow applies to wider Jordan-MPO
Hamiltonians built in this package.

## Load Packages

```julia
using TensorKit
using MPSKit
using MPSKitModels: FiniteChain
using DynamicalCorrelators
```

DynamicalCorrelators.jl reexports the model builders, operators, state helpers,
algorithm wrappers, and observable routines used below.

## Build a Hamiltonian

For common models, use the package model helpers:

```julia
N = 24
filling = (1, 1)

H = hubbard(Float64, SU2Irrep, U1Irrep, FiniteChain(N);
    t = 1.0,
    U = 8.0,
    μ = 0.0,
    filling = filling,
)
```

For custom Abelian models, construct the Hamiltonian through QuantumLattices and
then call `hamiltonian(...)`. That route is useful for arbitrary neighbor lists
and lattice geometries, while the model helpers are usually more convenient for
standard Hubbard, extended Hubbard, bilayer, Kitaev-Hubbard, and spin models.

## Create an Initial State

The symmetry-aware random-state constructors choose physical and virtual spaces
compatible with the filling:

```julia
ψ0 = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, N; filling)
```

For infinite or unit-cell calculations, use `randInfiniteMPS` with the same
symmetry and filling conventions.

## Find a Ground State

For a quick calculation, use the DMRG2 wrapper:

```julia
truncdims = [128, 256, 512]
gs, envs, ϵ = dmrg2(ψ0, H, truncdims; alg = myDMRG2())
E0 = expectation_value(gs, H, envs)
```

For larger sparse Jordan-MPO calculations, one-site CBE-DMRG is often the more
useful production path:

```julia
gs, envs, ϵ = dmrg1_cbe(ψ0, H, truncdims;
    cbe_method = :direct,
    filename = "dmrg1_cbe.jld2",
)
```

`dmrg1_cbe!` mutates the input state; `dmrg1_cbe` works on a copy.

## Choose Operators

Define a single local operator for the response:

```julia
sp = S_plus(Float64, SU2Irrep, U1Irrep; filling)
```

## Compute Real-Time Correlators

Use `dcorrelator` for zero-temperature dynamical correlations. In v0.11, the
recommended single-site long-time algorithm is CBE-TDVP1:

```julia
times = 0:0.05:10
tdvp_cbe = myTDVP1_CBE(D = 512)

gf_spin = dcorrelator(gs, H, sp, 1:N;
    times,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_spin",
)
```

For one source channel only, pass an integer `id` instead of an array:

```julia
gf_spin_site = dcorrelator(gs, H, sp, div(N, 2);
    times,
    record_indices = 1:101,
    tdvp1 = tdvp_cbe,
    gf_path = "gf_spin_site",
)
```

Integer source ids `1:N` and `N+1:2N` select the two conjugate channels for
single-operator correlators.

## Transform to Spectra

Use `fourier_kw` for momentum-frequency data and `fourier_rw` for real-space
frequency data:

```julia
rs = [[Float64(i)] for i in 1:N]
ks = [[k] for k in range(-pi, pi; length = 101)]
ws = range(-10, 10; length = 401)

gf_kw = fourier_kw(gf_spin, rs, times, ks, ws; broadentype = (0.05, "G"))
```

The spectral tutorials discuss broadening choices and multi-orbital regrouping.
