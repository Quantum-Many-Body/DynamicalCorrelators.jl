# Algorithms

All core algorithms (DMRG, TDVP, IDMRG) are provided by
[MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl), including Controlled
Bond Expansion through the `alg_expand` keyword of `DMRG` and `TDVP`
(`OptimalExpand`, `SketchedExpand`, `RandExpand`). The functions below are
convenience constructors for the package's default algorithm configurations;
drive them through MPSKit's `find_groundstate`, `timestep`/`timestep!`, and
`time_evolve`.

## DMRG drivers with bond-dimension schedules

```@docs
dmrg1!
dmrg1
dmrg2!
dmrg2
dmrg_mix!
dmrg_mix
```

## Default Algorithm Configurations

```@docs
myDMRG1
myDMRG2
myDMRG1_CBE
myTDVP1
myTDVP1_CBE
myTDVP2
```

## Fast Finite Engine

The finite-system drivers run on the package's lazy, optionally disk-backed
environment manager; see [The Finite Engine](../tutorials/finite_engine.md)
for the memory model and threading layout.

```@docs
FastFiniteEnvironments
fast_timestep!
configure_finite_engine!
set_threaded_hamiltonian!
set_prefused_hamiltonian!
env_memory_bytes
```

## Cluster Perturbation Theory (CPT)

```@docs
Perioder
CPT
singleParticleGreenFunction
spectrum
densityofstates
```
