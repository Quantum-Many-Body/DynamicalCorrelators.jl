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
dmrg1_cbe!
dmrg1_cbe
```

## Infinite DMRG

```@docs
idmrg2
```

## Default Algorithm Configurations

```@docs
myDMRG
myDMRG2
myTDVP
myTDVP1_CBE
myTDVP2
TDVP1_CBE
```

## Cluster Perturbation Theory (CPT)

```@docs
Perioder
CPT
singleParticleGreenFunction
spectrum
densityofstates
```
