# Getting Started

This tutorial covers the basic workflow for using DynamicalCorrelators.jl.

## Prerequisites

Make sure you have the following packages installed:

```julia
using Pkg
Pkg.add("DynamicalCorrelators")
```

## Basic Workflow

A typical calculation follows these steps:

1. **Define a lattice** — choose the geometry and boundary conditions
2. **Build a Hamiltonian** — specify the model parameters
3. **Find the ground state** — using DMRG
4. **Compute observables** — static/dynamical correlations, spectral functions

## Step 1: Lattice Construction

### Using pre-defined lattices from MPSKitModels

For simple 1D chains:

```julia
using MPSKitModels: FiniteChain, InfiniteChain

lattice = FiniteChain(48)   # 48-site open chain
```

### Using CustomLattice for 2D systems

For 2D systems, use the `Square` or `BilayerSquare` lattice types:

```julia
using DynamicalCorrelators

# 4×4 square lattice with open boundaries
sq = Square(4, 4)

# 2×2 bilayer square lattice with 2 orbitals per site
bl = BilayerSquare(2, 2; norbit=2)

# Custom lattice from QuantumLattices
using QuantumLattices
unitcell = Lattice([0.0, 0.0]; vectors=[[1, 0], [0, 1]])
lattice = Lattice(unitcell, (4, 4), ('o', 'o'))
cl = Custom(lattice)
```

## Step 2: Hamiltonian Construction

### Pre-defined Hamiltonians (with non-Abelian symmetry)

```julia
using TensorKit, MPSKit

filling = (1, 1)  # half-filling: P/Q = 1/1

# SU(2)×U(1) Hubbard model on a chain
H = hubbard(Float64, SU2Irrep, U1Irrep, FiniteChain(48);
            t=1.0, U=8.0, μ=0.0, filling=filling)
```

### From QuantumLattices (Abelian symmetry only)

```julia
using QuantumLattices

unitcell = Lattice([0.0, 0.0]; vectors=[[1, 0], [0, 1]])
lattice = Lattice(unitcell, (4, 4), ('o', 'o'))
hilbert = Hilbert(site => Fock{:f}(1, 2) for site in 1:length(lattice))

t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 8.0)

H = hamiltonian((t, U), lattice, hilbert; neighbors=1)
```

## Step 3: Ground State Search

```julia
# Create random initial MPS
st = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, 48; filling=filling)

# Find ground state with DMRG
gs, envs, delta = find_groundstate(st, H, DMRG2(trscheme=trunctol(1e-6)))

# Check ground state energy
E0 = expectation_value(gs, H)
println("Ground state energy: ", E0)
```

## Step 4: Compute Observables

### Static correlations

```julia
# Spin-spin correlation
sp = S_plus(Float64, SU2Irrep, U1Irrep; filling=filling)
corr = TwoSiteCorrelation((sp, sp), Custom(lattice), 1)
Fr = correlator(corr, gs)
```

### Dynamical correlations (see next tutorials)

```julia
cp = e_plus(Float64, SU2Irrep, U1Irrep; side=:L, filling=filling)
cm = e_min(Float64, SU2Irrep, U1Irrep; side=:L, filling=filling)
gf = dcorrelator(gs, H, (cp, cm); times=0:0.05:10, trscheme=truncrank(200))
```
