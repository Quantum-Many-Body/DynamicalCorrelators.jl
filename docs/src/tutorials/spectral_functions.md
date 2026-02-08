# Spectral Functions

This tutorial shows how to obtain momentum-frequency resolved spectral functions
from real-space, real-time correlation functions via Fourier transforms.

## Theory

The spectral function is obtained from the double Fourier transform of the Green's function:

```math
G(k, \omega) = \frac{1}{(2\pi)^2} \int_0^{t_\mathrm{end}} dt \int_0^L dx \, G(x, t) \, e^{-i(kx - \omega t)}
```

The spectral function is then:

```math
A(k, \omega) = -\frac{1}{\pi} \mathrm{Im}\, \mathrm{tr}\, G(k, \omega)
```

## Full workflow

```julia
using TensorKit, MPSKit, DynamicalCorrelators

# 1. Compute ground state
filling = (1, 1)
N = 48
H = hubbard(Float64, SU2Irrep, U1Irrep, FiniteChain(N); t=1, U=8, filling=filling)
st = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, N; filling=filling)
gs, _, _ = find_groundstate(st, H, DMRG2(trscheme=trunctol(1e-6)))

# 2. Compute G(r, t)
cp = e_plus(Float64, SU2Irrep, U1Irrep; side=:L, filling=filling)
cm = e_min(Float64, SU2Irrep, U1Irrep; side=:L, filling=filling)
ts = 0:0.05:20
gf_rt = dcorrelator(gs, H, (cp, cm); times=ts, trscheme=truncrank(200))

# 3. Fourier transform to G(k, ω)
# Define k-points and frequency range
ks = [[k] for k in range(-pi, pi, 100)]
ws = range(-10, 10, 500)

# Site positions (1D chain)
rs = [[Float64(i)] for i in 1:N]

# Double Fourier transform with Gaussian broadening (η = 0.05)
gf_kw = fourier_kw(gf_rt, rs, ts, ks, ws; broadentype=(0.05, "G"))

# 4. Extract spectral function A(k, ω) = -Im(tr(G))/π
A = -imag.(gf_kw) ./ π
```

## Broadening Options

Several broadening/windowing functions are available to suppress truncation artifacts:

| Type | Code | Formula |
|------|------|---------|
| Gaussian | `"G"` | ``e^{-(\eta t)^2}`` |
| Lorentzian | `"L"` | ``e^{-\eta|t|}`` |
| Blackman | `"B"` | ``0.42 - 0.5\cos(2\pi t/T) + 0.08\cos(4\pi t/T)`` |
| Parzen | `"P"` | Piecewise cubic taper |

Usage:

```julia
# Gaussian broadening with η = 0.05
gf_kw = fourier_kw(gf_rt, rs, ts, ks, ws; broadentype=(0.05, "G"))

# Lorentzian broadening with η = 0.1
gf_kw = fourier_kw(gf_rt, rs, ts, ks, ws; broadentype=(0.1, "L"))
```

## Real-space frequency transform

For computing the local density of states or real-space resolved spectral functions:

```julia
ws = range(-10, 10, 500)
gf_rw = fourier_rw(gf_rt, ts, ws; broadentype=(0.05, "G"))
```

## Static Structure Factor

For equal-time correlations, compute the static structure factor directly:

```julia
# ss is the real-space correlation matrix ⟨S_a S_b⟩
sf = static_structure_factor(ss, rs, ks)
```

## Multi-orbital Systems

For multi-orbital systems (e.g., bilayer), use the `regroup` parameter to specify orbital grouping:

```julia
# 2 orbitals per site, 8 sites → indices [1,2], [3,4], ...
regroup = [[2i-1, 2i] for i in 1:8]
gf_kw = fourier_kw(gf_rt, rs, ts, ks, ws; regroup=regroup)
```
