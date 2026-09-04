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
myDMRG
myDMRG2
myDMRG1_CBE
myTDVP
myTDVP1_CBE
myTDVP2
```

## Cluster Perturbation Theory (CPT)

```@docs
Perioder
CPT
singleParticleGreenFunction
spectrum
densityofstates
```
