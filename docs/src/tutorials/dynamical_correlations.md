# Dynamical Correlations

This page describes zero-temperature real-time correlation functions computed
with TDVP time evolution.

The core idea is simple: apply a local operator to the ground state, evolve that
charged state in real time, and measure overlaps with operator-applied ground
states at each time step.

## Single-Operator Real-Time Correlators

`dcorrelator` evolves one source operator at a time. For an operator `O`, the
zero-temperature workflow builds charged states `O_j|gs>` and measures all
overlaps against operator-applied ground states during TDVP evolution:

```math
G_{ij}(t) =
-i\langle gs|O_i^\dagger e^{-iHt} O_j|gs\rangle .
```

Pass a single operator and either one source id or a collection of source ids:

```julia
using TensorKit
using MPSKit
using MPSKitModels: FiniteChain
using DynamicalCorrelators

N = 24
filling = (1, 1)
H = hubbard(Float64, SU2Irrep, U1Irrep, FiniteChain(N);
    t = 1.0, U = 8.0, filling = filling)

ψ0 = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, N; filling)
gs, envs, ϵ = dmrg2(ψ0, H, [128, 256, 512]; alg = myDMRG2())

times = 0:0.05:20
tdvp_cbe = myTDVP1_CBE(D = 512)
sp = S_plus(Float64, SU2Irrep, U1Irrep; filling)

gf = dcorrelator(gs, H, sp, 1:N;
    times,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_spin",
)
```

## One-Operator Responses

For spin, charge, or other one-operator responses, pass a single operator and
one source id or a set of source ids:

```julia
sp = S_plus(Float64, SU2Irrep, U1Irrep; filling)

gf_spin = dcorrelator(gs, H, sp, 1:N;
    times,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_spin",
)
```

The result has dimensions

```julia
(length(H), length(indices), length(record_indices))
```

where the first dimension is the measured site and the second dimension is the
source channel.

## Single-Source Calculations

When only one source site is needed, pass an integer id. This avoids
`SharedArray` allocation and distributed scheduling:

```julia
site = div(N, 2)

gf_site = dcorrelator(gs, H, sp, site;
    times,
    record_indices = 1:101,
    tdvp1 = tdvp_cbe,
    gf_path = "gf_spin_site",
)
```

For single-operator correlators, ids `1:N` and `N+1:2N` select the two conjugate
source channels.

## Choosing TDVP Algorithms

The standard pattern is:

```julia
dcorrelator(gs, H, op, indices;
    times,
    n = 3,
    tdvp1 = myTDVP(),
    tdvp2 = myTDVP2(; trunc=truncrank(512)),
)
```

Here the first `n` steps use two-site TDVP and later steps use one-site TDVP.
This is MPSKit's conventional warmup strategy.

In v0.11, CBE-TDVP1 is often the preferred long-time path:

```julia
tdvp_cbe = myTDVP1_CBE(D = 512, delta = 0.1, cbe_tol = 1e-10)

dcorrelator(gs, H, op, indices;
    times,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
)
```

With this choice, bond growth is controlled by `D` in the CBE algorithm rather
than by the `trunc` setting of `myTDVP2`.

## Checkpointing

Each source channel is saved to a JLD2 file under `gf_path`. Re-running the same
calculation loads completed files and recomputes incomplete ones:

```julia
gf = dcorrelator(gs, H, sp, 1:N;
    times = 0:0.05:20,
    record_indices = 1:201,
    gf_path = "gf_spin",
    tdvp1 = tdvp_cbe,
)
```

Use separate `gf_path` directories for calculations with different operators,
time grids, or TDVP algorithms.

## What `sweep_dot` Does

For zero-temperature single-operator correlators, `dcorrelator` uses `sweep_dot`
to compute all overlaps

```math
\langle gs|O_i^\dagger|\psi(t)\rangle
```

in one left/right environment sweep. Operators with `(1,2)` legs in the
`side=:L` convention and charge-neutral `(1,1)` operators are supported.
