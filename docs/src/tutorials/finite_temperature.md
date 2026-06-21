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
    tdvp1 = myTDVP(),
    tdvp2 = myTDVP2(; trunc=truncrank(256)),
    n = 3,
)
```

Then evolve the prepared purification in real time and save every `rho(t)`:

```julia
times = 0:0.05:10
rho_path = "rho_beta_2_realtime.jld2"
evolve_mps(H, times, rhoβ;
    filename = rho_path,
    save_id = eachindex(times),
    tdvp1 = myTDVP(),
    tdvp2 = myTDVP2(; trunc=truncrank(256)),
    n = 3,
)
```

## Single-Operator Finite-T Correlators

For one source channel, pass an integer `id`:

```julia
sp = S_plus(Float64, SU2Irrep, U1Irrep; filling)
tdvp_cbe = myTDVP1_CBE(D = 256)

gf_site = dcorrelator(rho_path, H, sp, div(N, 2);
    times,
    beta = β,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_finiteT_site",
)
```

This method evolves only the charged source ket in memory and loads the
corresponding `rho(t)` from `rho_path` for each time slice. It returns an array
of size `(length(H), length(times))`.

## Multi-Source Finite-T Correlators

To compute many source channels in one call:

```julia
gf = dcorrelator(rho_path, H, sp, 1:N;
    times,
    beta = β,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_finiteT_spin",
)
```

Completed source files are loaded from `gf_path`; unfinished source channels are
evaluated independently, and distributed workers each keep one charged ket and
one loaded thermal state at a time.

## Notes

- `identityMPS(H)` constructs the purified infinite-temperature state.
- `dot(rho(0), rho(0))` is used internally as the partition-function
  normalization.
- The finite-temperature `dcorrelator` methods take `rho_path` as their first
  argument and expect keys of the form `"t=$(times[k])"`.
- CBE-TDVP1 is useful here for the same reason as at zero temperature: it lets
  the one-site time-evolution path expand bonds in a controlled way.
