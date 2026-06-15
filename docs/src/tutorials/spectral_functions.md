# Spectral Functions

This page starts from real-space, real-time correlators and transforms them to
momentum-frequency data.

For a one-dimensional system, the transform has the form

```math
G(k,\omega) =
\frac{1}{(2\pi)^2}
\int dt \int dx \, G(x,t) e^{-i(kx-\omega t)} .
```

On finite data, `fourier_kw` performs the corresponding discrete sum and can
apply a damping window to reduce finite-time ringing.

## Compute Real-Time Data

```julia
using TensorKit
using MPSKit
using MPSKitModels: FiniteChain
using DynamicalCorrelators

N = 32
filling = (1, 1)
H = hubbard(Float64, SU2Irrep, U1Irrep, FiniteChain(N);
    t = 1.0, U = 8.0, filling = filling)

ψ0 = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, N; filling)
gs, envs, ϵ = dmrg2(ψ0, H, [128, 256, 512]; alg = myDMRG2())

cp = e_plus(Float64, SU2Irrep, U1Irrep; side = :L, filling)
cm = e_min(Float64, SU2Irrep, U1Irrep; side = :L, filling)

times = 0:0.05:20
tdvp_cbe = myTDVP1_CBE(D = 512)

gf_rt = dcorrelator(gs, H, (cp, cm);
    times,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_electron",
)
```

For a spin response, replace the operator pair with a single operator and a set
of source ids:

```julia
sp = S_plus(Float64, SU2Irrep, U1Irrep; filling)
gf_rt_spin = dcorrelator(gs, H, sp, 1:N; times, tdvp1 = tdvp_cbe)
```

## Momentum-Frequency Transform

Define real-space positions, momenta, and frequencies:

```julia
rs = [[Float64(i)] for i in 1:N]
ks = [[k] for k in range(-pi, pi; length = 121)]
ws = range(-10, 10; length = 501)
```

Then transform:

```julia
gf_kw = fourier_kw(gf_rt, rs, times, ks, ws; broadentype = (0.05, "G"))
```

For a fermionic spectral function, a common postprocessing step is:

```julia
Akw = -imag.(gf_kw) ./ pi
```

The exact sign and trace convention depends on how the Green's function was
assembled.

## Damping Windows

Finite-time data usually needs broadening or windowing:

| Code | Window |
|------|--------|
| `"G"` | Gaussian |
| `"L"` | Lorentzian |
| `"B"` | Blackman |
| `"P"` | Parzen |

Examples:

```julia
gf_kw_gaussian = fourier_kw(gf_rt, rs, times, ks, ws;
    broadentype = (0.05, "G"))

gf_kw_lorentzian = fourier_kw(gf_rt, rs, times, ks, ws;
    broadentype = (0.10, "L"))
```

Choose a damping scale comparable to the desired frequency resolution. Stronger
damping suppresses ringing but broadens peaks.

## Real-Space Frequency Data

For local density-of-states style calculations, transform only in time:

```julia
gf_rw = fourier_rw(gf_rt, times, ws; broadentype = (0.05, "G"))
```

## Multi-Orbital or Clustered Sites

For systems where several MPS sites should be treated as one physical unit, pass
`regroup`:

```julia
regroup = [[2i - 1, 2i] for i in 1:(N ÷ 2)]
gf_kw_orbital = fourier_kw(gf_rt, rs, times, ks, ws;
    regroup = regroup,
    broadentype = (0.05, "G"),
)
```

Use this for bilayer or multi-band layouts where the MPS ordering contains
several orbitals per unit cell.

## Static Structure Factor

For equal-time correlations:

```julia
Sk = static_structure_factor(ss, rs, ks)
```

where `ss` is the real-space static correlation matrix.
