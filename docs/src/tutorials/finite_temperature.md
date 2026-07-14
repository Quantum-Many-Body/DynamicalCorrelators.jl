# Finite-Temperature Correlations

Finite-temperature calculations use purification: the density matrix is treated
as a pure MPS in a doubled Hilbert space. In this package that state has type
`FiniteSuperMPS`.

```math
|\rho_\beta\rangle =
\exp(-\beta H / 2)|\mathbb{I}\rangle .
```

Here ``|\mathbb{I}\rangle`` is the vectorized infinite-temperature density
matrix. If ``\rho : \mathcal{H}\to\mathcal{H}`` is regarded as an operator, the
purified state ``|\rho\rangle`` lives in
``\mathcal{H}\otimes\mathcal{H}^*``. The physical Hilbert-space leg is evolved
by ``H`` while the auxiliary leg stores the second index of the density matrix.

For a fermionic Green's function, the finite-temperature trace can be written
symmetrically as

```math
G_{ij}(\beta,t) =
\frac{
\mathrm{Tr}\left(
e^{-\beta H/2} e^{iHt} c_i e^{-iHt} c_j^\dagger e^{-\beta H/2}
\right)}
{\mathrm{Tr}\left(e^{-\beta H}\right)} .
```

After vectorization this becomes an overlap between two pure states,

```math
G_{ij}(\beta,t) =
\frac{
\left\langle
e^{-\beta H/2} e^{iHt} c_i
\middle|
e^{-iHt} c_j^\dagger e^{-\beta H/2}
\right\rangle}
{Z_\beta},
\qquad
Z_\beta = \langle \rho_\beta | \rho_\beta\rangle .
```

This is the formula implemented by the finite-temperature `dcorrelator`
methods. `identityMPS(H)` constructs ``|\mathbb{I}\rangle``. The first
`evolve_mps` call prepares ``|\rho_\beta\rangle`` from that state. The second
`evolve_mps` call saves the real-time trajectory
``|\rho_\beta(t)\rangle = e^{-iHt}|\rho_\beta\rangle`` in a JLD2 file. During
`dcorrelator`, `chargedMPS(op, rho, j)` builds the source state
``c_j^\dagger|\rho_\beta\rangle`` or its analogue for `op`, that source state
is evolved in real time, and `sweep_dot(rho_t, op, ket)` evaluates all sink
sites ``i`` at the current time slice. The normalization is computed as
`Z = dot(rho, rho)`.

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
of size `(length(H), length(times))`. The returned values include the package's
real-time Green-function prefactor `-im`, so the stored overlap is converted to
the same convention used by the zero-temperature `dcorrelator` interface.

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
- The `rho_path` trajectory must contain every time requested by `times`,
  because `dcorrelator` loads the saved ``|\rho_\beta(t)\rangle`` slice by
  slice instead of evolving it again.
- CBE-TDVP1 is useful here for the same reason as at zero temperature: it lets
  the one-site time-evolution path expand bonds in a controlled way.
