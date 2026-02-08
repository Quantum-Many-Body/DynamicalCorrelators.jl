# Algorithms

All core algorithms (DMRG, TDVP, IDMRG) are provided by
[MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl).
The functions below are convenience wrappers that add progress logging,
automatic JLD2 file saving, and multi-sweep truncation management.

## DMRG

```@docs
dmrg2!
dmrg2
dmrg2_sweep!
```

## Infinite DMRG

```@docs
idmrg2
```

## Default Algorithm Configurations

```@docs
DefaultDMRG
DefaultDMRG2
DefaultTDVP
DefaultTDVP2
```

## Cluster Perturbation Theory (CPT)

```@docs
Perioder
CPT
singleParticleGreenFunction
spectrum
densityofstates
```
