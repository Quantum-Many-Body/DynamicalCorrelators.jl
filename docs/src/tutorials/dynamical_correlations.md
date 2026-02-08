# Dynamical Correlations

This tutorial explains how to compute dynamical correlation functions using
TDVP time evolution (provided by [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl)).

## Theory

The retarded single-particle Green's function is defined as:

```math
G^R_{ij}(t) = -i\theta(t)\langle \psi_0 | \{c_i(t), c_j^\dagger(0)\} | \psi_0 \rangle
```

which can be decomposed into greater and lesser components:

```math
G^>_{ij}(t) = -i e^{iE_0 t} \langle \psi_0 | c_i \, e^{-iHt} \, c_j^\dagger | \psi_0 \rangle
```

The key computational step is evolving the "charged" MPS ``c_j^\dagger|\psi_0\rangle`` in time
using TDVP, then computing overlaps with all ``\langle\psi_0|c_i`` at each time step.

## Single-particle Green's function

### Using operator pair `(c†, c)`

```julia
using TensorKit, MPSKit, DynamicalCorrelators

filling = (1, 1)
H = hubbard(Float64, SU2Irrep, U1Irrep, FiniteChain(24); t=1, U=8, filling=filling)
st = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, 24; filling=filling)
gs, _, _ = find_groundstate(st, H, DMRG2(trscheme=trunctol(1e-6)))

# Define creation and annihilation operators
cp = e_plus(Float64, SU2Irrep, U1Irrep; side=:L, filling=filling)
cm = e_min(Float64, SU2Irrep, U1Irrep; side=:L, filling=filling)

# Compute retarded Green's function G(r, t)
# This evolves c†_j|gs⟩ and c_j|gs⟩ for all j, computing overlaps at each time step
gf = dcorrelator(gs, H, (cp, cm);
    times=0:0.05:20,        # time grid
    trscheme=truncrank(200), # bond dimension control
    n=3                      # use TDVP2 for first n steps, then TDVP1
)
```

### Using pre-computed charged MPS

For large systems, you may want to pre-compute and store the charged MPS:

```julia
# Pre-compute all c†_j|gs⟩
mps = [chargedMPS(cp, gs, j) for j in 1:length(H)]

# Then pass them directly
gf = dcorrelator(gs, H, cp, 1:length(H);
    times=0:0.05:20,
    trscheme=truncrank(200)
)
```

## Spin-spin dynamical correlations

For two-particle correlations like ``\langle S^+(t) S^-(0) \rangle``:

```julia
sp = S_plus(Float64, SU2Irrep, U1Irrep; filling=filling)

# Compute spin dynamical structure factor
gf_spin = dcorrelator(gs, H, sp, 1:length(H);
    times=0:0.1:50,
    trscheme=truncrank(200)
)
```

## Checkpoint and Resume

The `dcorrelator` function automatically saves intermediate results to JLD2 files in the
`gf_path` directory. If a computation is interrupted, it will resume from the last saved
checkpoint when re-run:

```julia
gf = dcorrelator(gs, H, (cp, cm);
    times=0:0.05:20,
    gf_path="./green_functions/",  # directory for checkpoints
    trscheme=truncrank(200)
)
```

## TDVP Algorithm Selection

The parameter `n` controls the switch from two-site TDVP (TDVP2) to single-site TDVP (TDVP1):

- **TDVP2** (first `n` steps): Allows bond dimension growth, more expensive but captures entanglement growth.
- **TDVP1** (remaining steps): Fixed bond dimension, faster but may miss entanglement.

For typical calculations, `n=2` or `n=3` is sufficient.
