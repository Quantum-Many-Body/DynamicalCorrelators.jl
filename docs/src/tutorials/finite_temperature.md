# Finite Temperature Calculations

This tutorial covers finite-temperature dynamical correlations using the
thermofield double (purification) approach.

## Theory

In the purification approach, the thermal density matrix ``\rho = e^{-\beta H}`` is represented
as a pure state in a doubled Hilbert space (physical + ancilla):

```math
|\rho\rangle = e^{-\beta H / 2} |\mathbb{I}\rangle
```

where ``|\mathbb{I}\rangle`` is the identity MPS (maximally entangled state between physical
and ancilla spaces). This `FiniteSuperMPS` has 3 physical legs per site
(virtual, physical, ancilla; virtual).

## Workflow

### Step 1: Construct the identity MPS

```julia
using TensorKit, MPSKit, DynamicalCorrelators

filling = (1, 1)
H = hubbard(Float64, SU2Irrep, U1Irrep, FiniteChain(12); t=1, U=8, filling=filling)

# Create identity MPS (infinite temperature state)
rho = identityMPS(H)
```

### Step 2: Evolve to finite temperature

Cool down to inverse temperature ``\beta`` by evolving in imaginary time using
TDVP (from [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl)):

```julia
beta = 2.0  # inverse temperature
ts_beta = 0:0.01:(beta/2)  # imaginary time steps (evolve β/2)

rho_beta = evolve_mps(H, ts_beta, rho;
    filename="rho_beta_$(beta).jld2",
    trscheme=truncrank(200),
    n=3
)
```

### Step 3: Compute finite-T correlations

```julia
cp = e_plus(Float64, SU2Irrep, U1Irrep; side=:L, filling=filling)

gf = dcorrelator(rho_beta, H, cp, 1:length(H);
    times=0:0.05:10,
    beta=beta,
    trscheme=truncrank(200)
)
```

## Key differences from zero temperature

1. **MPS type**: Uses `FiniteSuperMPS` (3-leg tensors) instead of `FiniteNormalMPS` (2-leg tensors).
2. **Operator application**: The `chargedMPS` function handles the extra ancilla leg automatically.
3. **Normalization**: The partition function ``Z = \langle\rho|\rho\rangle`` is computed internally.
4. **Time evolution**: Both the density matrix ``|\rho(t)\rangle`` and the charged states need to be evolved.
