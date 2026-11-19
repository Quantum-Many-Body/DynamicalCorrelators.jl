# Dynamical Correlations

This page describes zero-temperature real-time correlation functions computed
with TDVP time evolution. The same numerical primitive is used for spin
responses, normal fermionic Green's functions, and Gorkov Green's functions:
apply a source operator to the ground state, evolve that charged state, and
measure overlaps with operator-applied ground states during the time evolution.

Throughout this page, `|0>` is the ground state, `E0 = <0|H|0>`, and

```math
O_i(t) = e^{iHt} O_i e^{-iHt}.
```

`dcorrelator` stores data only for the real-time grid `times`, usually with
`times[1] == 0`.

## Source Channels

For a chain of length `N = length(H)`, a source id chooses both a site and one
of the greater/lesser source contributions:

| Source id | Site | Contribution | Phase used by `dcorrelator` |
|-----------|------|--------------|------------------------------|
| `j in 1:N` | `j` | greater | `exp(+im * E0 * t)` |
| `N+j` | `j` | lesser term in the retarded combination | `exp(-im * E0 * t)` and complex conjugation |

For a single local operator `op`, the greater source contribution evolves

```math
|\psi_j(t)\rangle = e^{-iHt} op_j |0\rangle
```

and computes

```math
-i e^{+iE_0t}
\langle op_i 0 | e^{-iHt} op_j 0\rangle,
```

where `|op_i 0>` denotes `op_i|0>`. The source id `N+j` computes the lesser
term in the retarded combination:

```math
-i e^{-iE_0t}
\overline{\langle op_i 0 | e^{-iHt} op_j 0\rangle}.
```

This second quantity is the lesser contribution with the sign used inside the
retarded anticommutator. With the common convention
`G^<(t) = +i <c_j^\dagger(0)c_i(t)>`, the value returned by this source channel
is `-G^<(t)`, so a retarded Green's function is assembled as
`G^>(t) - G^<(t)`.

## Basic One-Operator Correlators

For spin, charge, or other one-operator responses, pass a single operator and
one source id or a collection of source ids:

```julia
using TensorKit
using MPSKit
using MPSKitModels: FiniteChain
using DynamicalCorrelators

N = 24
filling = (1, 1)
H = hubbard(Float64, SU2Irrep, U1Irrep, FiniteChain(N);
    t = 1.0, U = 8.0, filling = filling)

psi0 = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, N; filling)
gs, envs, eps = dmrg2(psi0, H, [128, 256, 512])

times = 0:0.05:20
tdvp_cbe = myTDVP1_CBE(D = 512)
sp = S_plus(Float64, SU2Irrep, U1Irrep; filling)

gf_spin = dcorrelator(gs, H, sp, 1:N;
    times,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_spin",
)
```

For `indices::AbstractArray`, the result has size

```julia
(length(H), length(indices), length(record_indices))
```

The first dimension is the measured site `i`, the second is the source channel
`j`, and the third is time. For a single integer source id, the result has size

```julia
(length(H), length(record_indices))
```

and avoids distributed `SharedArray` scheduling.

## Normal Fermionic Green's Function

The normal retarded fermion Green's function is

```math
G_{ij}(t)
= -i \langle \{ c_i(t), c_j^\dagger(0) \} \rangle .
```

With `cre = c^\dagger` and `ann = c`, the two terms are

```math
G_{ij}(t)
= -i \left[
e^{+iE_0t}
\langle c_i^\dagger 0 | e^{-iHt} c_j^\dagger 0\rangle
+
e^{-iE_0t}
\overline{\langle c_i 0 | e^{-iHt} c_j 0\rangle}
\right].
```

The greater term is produced by the creation source channel:

```julia
cre = e_plus(Float64, SU2Irrep, U1Irrep; side = :L, filling)
ann = e_min(Float64, SU2Irrep, U1Irrep; side = :L, filling)

G_greater = dcorrelator(gs, H, cre, 1:N;
    times,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_cre_greater",
)
```

The lesser term in the retarded combination is produced by the annihilation
source channel:

```julia
G_lesser_term = dcorrelator(gs, H, ann, N+1:2N;
    times,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_ann_lesser",
)

G_normal = G_greater .+ G_lesser_term
```

This two-call structure is intentional: the two expensive TDVP evolutions are
`e^{-iHt} c_j^\dagger |0>` and `e^{-iHt} c_j |0>`. The overlaps for all
measured sites are then obtained in sweeps.

## Gorkov Green's Function

For Hamiltonians with pairing terms, particle number is not conserved and the
normal Green's function is only one block of the Nambu, or Gorkov, Green's
function:

```math
\mathcal{G}_{ij}(t)
= -i
\begin{pmatrix}
\langle \{ c_i(t), c_j^\dagger(0) \} \rangle &
\langle \{ c_i(t), c_j(0) \} \rangle \\
\langle \{ c_i^\dagger(t), c_j^\dagger(0) \} \rangle &
\langle \{ c_i^\dagger(t), c_j(0) \} \rangle
\end{pmatrix}.
```

The package labels the four matrix positions as `G11`, `G12`, `G21`, and
`G22`:

```math
\mathcal{G}_{ij}(t)
=
\begin{pmatrix}
G11_{ij}(t) & G12_{ij}(t) \\
G21_{ij}(t) & G22_{ij}(t)
\end{pmatrix}.
```

Written in terms of the two independently evolved source states, the blocks are

```math
\begin{aligned}
G11_{ij}(t)
&= -i \left[
e^{+iE_0t}
\langle c_i^\dagger 0 | e^{-iHt} c_j^\dagger 0\rangle
+
e^{-iE_0t}
\overline{\langle c_i 0 | e^{-iHt} c_j 0\rangle}
\right], \\
G12_{ij}(t)
&= -i \left[
e^{+iE_0t}
\langle c_i^\dagger 0 | e^{-iHt} c_j 0\rangle
+
e^{-iE_0t}
\overline{\langle c_i 0 | e^{-iHt} c_j^\dagger 0\rangle}
\right], \\
G21_{ij}(t)
&= -i \left[
e^{+iE_0t}
\langle c_i 0 | e^{-iHt} c_j^\dagger 0\rangle
+
e^{-iE_0t}
\overline{\langle c_i^\dagger 0 | e^{-iHt} c_j 0\rangle}
\right], \\
G22_{ij}(t)
&= -i \left[
e^{+iE_0t}
\langle c_i 0 | e^{-iHt} c_j 0\rangle
+
e^{-iE_0t}
\overline{\langle c_i^\dagger 0 | e^{-iHt} c_j^\dagger 0\rangle}
\right].
\end{aligned}
```

The `anomalous` keyword lets one TDVP source evolution produce all four Gorkov
matrix-position contributions associated with that source. If the main operator
is `cre`, pass `anomalous = ann`:

`anomalous` is not inferred automatically: pass the opposite Nambu operator
explicitly. For pairing Hamiltonians, also choose MPS and operator symmetries
compatible with the Hamiltonian, for example fermion parity rather than a
particle-number `U(1)` sector when particle number is broken.

```julia
g_cre = dcorrelator(gs, H, cre, 1:N;
    anomalous = ann,
    times,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_gorkov_cre",
)
```

This evolves only `e^{-iHt} c_j^\dagger |0>`. At each time, it measures both
`cre` and `ann` projections, returning

```julia
g_cre.G11
g_cre.G12
g_cre.G21
g_cre.G22
```

For `indices = 1:N`, each array has size

```julia
(length(H), length(indices), length(record_indices))
```

The complementary source evolution is obtained by swapping the operators and
using ids `N+1:2N`:

```julia
g_ann = dcorrelator(gs, H, ann, N+1:2N;
    anomalous = cre,
    times,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
    gf_path = "gf_gorkov_ann",
)
```

A complete retarded Gorkov anticommutator is then assembled block by block:

```julia
G11 = g_cre.G11 .+ g_ann.G11
G12 = g_cre.G12 .+ g_ann.G12
G21 = g_cre.G21 .+ g_ann.G21
G22 = g_cre.G22 .+ g_ann.G22
```

This organization avoids repeated TDVP work. The only independent source
evolutions are `c_j^\dagger |0>` and `c_j |0>`; the four Gorkov positions are
formed from cheap projection sweeps and phase/conjugation factors.

## Choosing TDVP Algorithms

The standard pattern is:

```julia
dcorrelator(gs, H, op, indices;
    times,
    n = 3,
    tdvp1 = myTDVP1(),
    tdvp2 = myTDVP2(; trunc = truncrank(512)),
)
```

The first `n` steps use two-site TDVP and later steps use one-site TDVP. In
long-time calculations, CBE-TDVP1 is often the preferred path:

```julia
tdvp_cbe = myTDVP1_CBE(D = 512, delta = 0.1)

dcorrelator(gs, H, op, indices;
    times,
    tdvp1 = tdvp_cbe,
    tdvp2 = tdvp_cbe,
)
```

With this choice, bond growth is controlled by `D` in the CBE algorithm.

The time evolution runs in place on the fast finite engine, so its
environments follow the lazy `N+O(1)` memory model and can be offloaded to
disk with `disk = true` (or a directory path) at large bond dimensions; see
[The Finite Engine](finite_engine.md).

## Time-Split Scheme and Production Runs

All methods above evolve one charged ket through the full time grid — the
direct scheme. For production-size runs, `dcorrelator(gs, H, op; block, ...)`
(a three-positional-argument call with the required `block` keyword) selects
the time-split (meet-in-the-middle) scheme instead: both sides of the
operator sandwich are evolved to `t/2`, so every state is only evolved up to
`tmax/2`, halving the peak entanglement and the accumulated truncation error:

```julia
gf_greater = dcorrelator(gs, H, cp; block = :greater, times, ...)
gf_lesser  = dcorrelator(gs, H, cm; block = :lesser,  times, ...)
gfxt = gf_greater .+ gf_lesser
```

The phase/conjugation conventions match the direct method exactly, so the
retarded combination is assembled as before. Each normal call computes one
block and the full `length(H) × length(H)` matrix of site pairs at every
recorded slice; the two blocks are fully independent and can run as separate jobs.

For a full Nambu Green's function, evolve the creation and annihilation
families together:

```julia
gf_gorkov = dcorrelator(gs, H, cp;
    block = :gorkov, anomalous = cm, times, ...)
```

This returns a `2length(H) × 2length(H) × length(times)` array. Normal and
anomalous overlaps reuse the same checkpoints, so no full-time anomalous source
calculation is needed. The block phases and signs match the existing LNO
direct-source assembly: `G11 = gc.G11 + ga.G11`,
`G12 = gc.G12 - ga.G12`, `G21 = -gc.G21 + ga.G21`, and
`G22 = gc.G22 + ga.G22`.

The `dcorrelator` call above is the in-process threaded implementation. For
production runs, use one independent process per state instead:

```julia
dcorrelator_state(gs, H, cp;
    state = 1, block = :gorkov, anomalous = cm,
    times, workdir = "/shared/scratch/gorkov",
    n = 3, approxalg, tdvp1, tdvp2)
```

A normal block has `2L` states; a Gorkov calculation has `4L`. Gorkov state IDs
are creation bras `1:L`, annihilation bras `L+1:2L`, creation kets `2L+1:3L`,
and annihilation kets `3L+1:4L`. Each job resumes from its own latest
checkpoint and never waits for another state or for the overlap job.

Run one standalone overlap job on the same shared directory:

```julia
gf = dcorrelator_overlap("/shared/scratch/gorkov";
    gsenergy = E0, gf_path = "results", retention = 2)
```

It polls the atomic checkpoint files, computes every newly complete slice, and
saves the result under `meta/`. Only this job deletes old checkpoints; the
latest two slices of every state remain available for resume. There is no
master process, node-placement logic, or scheduling service. See the worked
SLURM example in [Basic Workflow on HPC](../index.md#Basic-Workflow-on-HPC).

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

Normal calculations save files of the form

```text
gf_start=<t_start>_end=<t_end>_id=<id>.jld2
```

Gorkov calculations with `anomalous` save separate files:

```text
gf_start=<t_start>_end=<t_end>_id=<id>_gorkov.jld2
```

Use separate `gf_path` directories for calculations with different operators,
time grids, or TDVP algorithms.

## What `sweep_dot` Does

For zero-temperature calculations, `dcorrelator` uses `sweep_dot` to compute all
overlaps

```math
\langle 0 | op_i^\dagger |\psi_j(t)\rangle
=
\langle op_i 0 | \psi_j(t)\rangle
```

in one left/right environment sweep over the measured site `i`. In Gorkov mode,
the same evolved source state is swept once with `op` and once with
`anomalous`. Operators with `(1,2)` legs in the `side = :L` convention and
charge-neutral `(1,1)` operators are supported.
