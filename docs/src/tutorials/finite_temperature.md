# Finite-Temperature Correlations

Finite-temperature calculations use purification: the density matrix is treated
as a pure MPS in a doubled Hilbert space. In this package that state has type
`FiniteSuperMPS`.

```math
|\rho_\beta\rangle =
\exp(-\beta H / 2)|\mathbb{I}\rangle .
```

## Prepare the Purification

Start from the infinite-temperature identity MPS:

```julia
using TensorKit
using MPSKit
using MPSKitModels: FiniteChain
using DynamicalCorrelators

N = 12
filling = (1, 1)
H = hubbard(Float64, SU2Irrep, U1Irrep, FiniteChain(N);
    t = 1.0, U = 8.0, filling = filling)

rho0 = identityMPS(H)
```

Cool the state by imaginary-time evolution to `β/2`:

```julia
β = 2.0
τs = 0:0.01:(β / 2)

rhoβ = evolve_mps(H, τs, rho0;
    filename = "rho_beta_2.jld2",
    tdvp1 = myTDVP,
    tdvp2 = myTDVP2(truncrank(256)),
    n = 3,
)
```

The saved file contains selected purification states from the imaginary-time
path. The real-time finite-temperature correlator below does not need a cached
array of all `rho(t)` states.

## Single-Operator Finite-T Correlators

For one source channel, pass an integer `id`:

```julia
sp = S_plus(Float64, SU2Irrep, U1Irrep; filling)
times = 0:0.05:10
tdvp_cbe = myTDVP1_CBE(D = 256)

gf_site = dcorrelator(rhoβ, H, sp, div(N, 2);
    times,
    beta = β,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_finiteT_site",
)
```

This method evolves only the current purification `rho_t`, the charged source
ket, and their environments. It returns an array of size
`(length(H), length(times))`.

## Multi-Source Finite-T Correlators

To compute many source channels in one call:

```julia
gf = dcorrelator(rhoβ, H, sp, 1:N;
    times,
    beta = β,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_finiteT_spin",
)
```

Completed source files are loaded from `gf_path`; active sources are evolved
together. This avoids the older memory-heavy strategy of precomputing all
thermal states and all charged bra states for every time point.

## Pair-Operator Finite-T Correlators

Fermionic Green's functions use an operator pair:

```julia
cp = e_plus(Float64, SU2Irrep, U1Irrep; side = :L, filling)
cm = e_min(Float64, SU2Irrep, U1Irrep; side = :L, filling)

gf_electron = dcorrelator(rhoβ, H, (cp, cm);
    times,
    beta = β,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_finiteT_electron",
)
```

The two channel groups are combined as a fermionic sum by default. Set
`isfermion=false` for the opposite sign convention.

## Notes

- `identityMPS(H)` constructs the purified infinite-temperature state.
- `dot(rhoβ, rhoβ)` is used internally as the partition-function
  normalization.
- The `rho_path` keyword is accepted by finite-temperature `dcorrelator`
  methods for compatibility, but v0.11 does not use cached `rho(t)` trajectory
  files in real-time correlators.
- CBE-TDVP1 is useful here for the same reason as at zero temperature: it lets
  the one-site time-evolution path expand bonds in a controlled way.
