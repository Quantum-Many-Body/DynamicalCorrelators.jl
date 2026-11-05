# DynamicalCorrelators.jl

DynamicalCorrelators.jl is a frontend for matrix-product-state calculations of
ground states, real-time dynamical correlation functions, finite-temperature
correlators, and momentum-frequency spectral functions.

The package is built around three layers:

- symbolic lattice and model construction through
  [QuantumLattices.jl](https://github.com/Quantum-Many-Body/QuantumLattices.jl)
  and model helpers in this package;
- finite and infinite MPS algorithms from
  [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl);
- convenience workflows for charged states, TDVP time evolution, checkpointed
  dynamical correlators, and Fourier transforms.

## Core Features

DynamicalCorrelators.jl provides convenience wrappers for the common pieces of
finite-MPS dynamical-correlation workflows:

- `myDMRG1_CBE` configures one-site DMRG with Controlled Bond Expansion (CBE)
  through MPSKit's `DMRG(; alg_expand = CBEExpand(...))`.
- `myTDVP1_CBE` configures CBE-assisted one-site TDVP through MPSKit's
  `TDVP(; alg_expand = CBEExpand(...), trunc = ...)`. This lets single-site
  TDVP grow bonds through CBE while keeping the cheaper one-site time-evolution
  sweep.
- `excited_dmrg1!`/`excited_dmrg2!` compute low-lying excited states by the
  penalty method on the same fast engine.
- `dcorrelator` supports single-source and multi-source checkpointed
  real-time correlator calculations, plus a time-split (meet-in-the-middle)
  scheme with a distributed two-job workflow (`prepare_dcorrelator` +
  `mode = :distributed`) for production-size runs.
- Static correlators are computed with a lazy `CorrelationPlan`: channels
  register geometry only, and all raw expectation values are evaluated in one
  shared, threaded pass with automatic reuse between channels.
- finite-temperature correlators read a saved `rho(t)` trajectory one slice at
  a time and use sweep contractions against the active charged ket.

## Basic Workflow on HPC

A production zero-temperature run splits into separate jobs submitted through
SLURM (`sbatch`): one DMRG job for the ground state, then a preparation job
and a lean follow-up job per operator block for the dynamical correlator.

### Ground state (`gs.jl`)

```julia
using Pkg
Pkg.activate("$(ENV["HOME"])/envs/userenv/")   # the project environment with
                                               # DynamicalCorrelators & co.

using TensorKit
using MPSKit
using DynamicalCorrelators
using QuantumLattices
using JLD2: save, load

# the engine parallelizes above BLAS (Hamiltonian channels, SVD blocks),
# so multi-threaded BLAS only adds contention
configure_finite_engine!(; blas_threads = 1, verbose = true)
set_threaded_hamiltonian!(Threads.nthreads())

# 6x6 square Hubbard with SU(2) spin x U(1) particle symmetry
coords = snake_2D([[1.0, 0.0], [0.0, 1.0]], vcat([[2,2,2,2,2,1,-2,-2,-2,-2,-2,1] for _ in 1:3]...)[1:end-1])
lattice = Lattice(coords...)
elt = Float64
t, u, filling = 1.0, 8.0, (1, 1)
H = hubbard(elt, SU2Irrep, U1Irrep, Custom(lattice); t=t, U=u, filling=filling)

tag = "hubbard_L=$(length(H))_t=$(t)_U=$(u)"
ckpt = "/bbfs/fsa/username/jobname/$(tag).jld2"
env_disk = "/bbfs/scratch/username/jobname"

# random initial state in the target charge sector; md is the (small)
# starting bond dimension — the schedule grows it
ψ = randFiniteMPS(elt, SU2Irrep, U1Irrep, length(H); md=20, filling=filling)

# Phase 1: two-site warmup ramp, one sweep per truncdims entry
# (no tail sweeps here — they run in phase 2 at the final D)
trunc2 = [64, 1024, 4096, 8192]
ψ, envs, E0 = dmrg2!(ψ, H, trunc2; max_tailsweeps=0,
    filename=ckpt,      # rolling checkpoint (keys "ψ", "E", ...) + mid-sweep
                        # update-level resume after a kill
    disk=env_disk,      # disk-backed environments (point at node-local
                        # scratch or a burst buffer, not NFS)
    verbose=2)

# Phase 2: one-site CBE sweeps at the final D until |ΔE| <= etol.
# Reloading from the checkpoint means this phase can also run as a
# separate job (e.g. after phase 1 finished or crashed):
ψ = load(ckpt, "ψ")
envs = HalfFiniteEnvironments(ψ, H; disk=env_disk)
ψ, envs, E0 = dmrg1!(ψ, H, [8192]; etol=1e-8, envs,
    max_tailsweeps=20, filename=ckpt, verbose=2)
```

Submit the whole node to a single multi-threaded Julia process with SLURM
(`gs.slurm`):

```bash
#!/bin/bash
#SBATCH -J gs
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 36
#SBATCH --mem=0
#SBATCH -o output/gs_%j

cd "$SLURM_SUBMIT_DIR"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export JULIA_EXCLUSIVE=1

julia -t 36 gs.jl
```

```bash
sbatch gs.slurm
```

### Dynamical correlation (`prep.jl` + `gf.jl`)

Production runs use the time-split (meet-in-the-middle) correlator: both
sides of the operator sandwich are evolved to `t/2`, halving the peak
entanglement and the accumulated truncation error compared to evolving one
ket through the full time grid. The run is split into two jobs by memory
profile — a short, generously provisioned **preparation job** (charged
states + the two-site TDVP window = the memory high-water mark), then a
lean **follow-up job** that resumes every state from its checkpoint and
runs the long one-site TDVP tail plus the overlaps. The greater block
(source `c†`) and the lesser block (source `c`) are fully independent and
run as separate array tasks.

Shared setup (`common_setup.jl`) — model, ground state, operator/block
choice, and the settings both jobs must agree on (`block`, `times`,
`record_indices`, `n`, guarded by `meta/config.jld2` under `workdir`):

```julia
using Pkg
Pkg.activate("$(ENV["HOME"])/envs/userenv/")

using TensorKit
using MPSKit
using DynamicalCorrelators
using QuantumLattices
using JLD2: save, load

i = parse(Int, ARGS[1])            # 1 = greater block (c†), 2 = lesser block (c)
elt = Float64
t, u, filling = 1.0, 8.0, (1, 1)
L = 36

# reload the converged ground state from the checkpoint written by gs.jl
gs = load("/bbfs/fsa/username/jobname/hubbard_L=$(L)_t=$(t)_U=$(u).jld2", "ψ")

coords = snake_2D([[1.0, 0.0], [0.0, 1.0]], vcat([[2,2,2,2,2,1,-2,-2,-2,-2,-2,1] for _ in 1:3]...)[1:end-1])
lattice = Lattice(coords...)
H = hubbard(elt, SU2Irrep, U1Irrep, Custom(lattice); t=t, U=u, filling=filling)

op, block = i == 1 ?
    (e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling), :greater) :
    (e_min(elt, SU2Irrep, U1Irrep;  side=:L, filling=filling), :lesser)

times = 0:0.1:100
workdir = "/bbfs/scratch/username/jobname/gf_split_$(block)/"
approxalg = myDMRG2(; tol = 1e-6, maxiter = 50, trunc = truncrank(8192))
tdvp2 = myTDVP2(; trunc = truncrank(8192))
```

Job 1, the preparation pass (`prep.jl`): computes the charged initial
states and evolves them only through the first `n` two-site TDVP steps;
no overlaps are computed. This is the memory high-water mark of the whole
run but only covers the first `n` timesteps, so give it **fat nodes and a
short walltime**. Idempotent — a rerun after a kill resumes each state
from its last checkpoint.

```julia
include("common_setup.jl")

using Distributed
using ClusterManagers

# preparation is short and memory-hungry: many single-threaded workers on
# one fat node to compress/evolve as many states as possible concurrently.
# For a multi-node preparation job, use addprocs(SlurmManager(np)) instead —
# the driver discovers the worker -> host mapping itself
addprocs(LocalAffinityManager(70; mode = COMPACT);
         exeflags = `--project=$(ENV["HOME"])/envs/userenv/ -t1`)

@everywhere begin
    using TensorKit
    using MPSKit
    using DynamicalCorrelators
    using QuantumLattices
    using JLD2: save, load
end

prepare_dcorrelator(gs, H, op;
    block, times, n = 3,
    approxalg, tdvp2,
    workdir,
    disk = "/bbfs/scratch/username",   # disk-backed environments
    charged_per_node = 4,              # node memory / per-compression peak:
                                       # lift the cap only on a warm
                                       # chargedmps_path cache
)
```

Job 2, the lean follow-up (`gf.jl`): resumes every state from its
slice-`r_stop` checkpoint, assembles the first overlap slices straight
from disk, and continues with the cheap one-site TDVP for the remaining
hundreds of timesteps. Its memory profile is set by `tdvp1` alone, so give
it **modest resources and a long walltime** — a different worker count is
fine, the driver discovers the worker → host mapping itself and resumes
from the checkpointed states.

```julia
include("common_setup.jl")

using Distributed
using ClusterManagers

# the follow-up is long and memory-light: fewer workers on a standard node
addprocs(LocalAffinityManager(32; mode = COMPACT);
         exeflags = `--project=$(ENV["HOME"])/envs/userenv/ -t1`)

@everywhere begin
    using TensorKit
    using MPSKit
    using DynamicalCorrelators
    using QuantumLattices
    using JLD2: save, load
end

gf = dcorrelator(gs, H, op;
    block, times, n = 3,
    approxalg, tdvp2,
    tdvp1 = myTDVP1(),
    mode = :distributed,
    workdir,                            # must match the preparation job
    gf_path = "/bbfs/fsa/username/jobname/gf_L=$(L)_U=$(u)_tmax=100/",
)

save("/bbfs/fsa/username/jobname/gf_L=$(L)_U=$(u)_tmax=100/gf_xt_$(block).jld2", "gf", gf)
```

The two SLURM scripts request different resources — the preparation job
gets fat nodes and a short limit, the follow-up a standard node and a long
one. `prep.slurm`:

```bash
#!/bin/bash
#SBATCH -J gf_prep
#SBATCH --array=1-2
#SBATCH -N 1                   # fat preparation job: one big-memory node
#SBATCH -n 1
#SBATCH -c 72                  # (use a bigmem partition here)
#SBATCH --mem=0
#SBATCH -t 12:00:00            # short: only the charged states + n tdvp2 steps
#SBATCH -o output/prep_%A_%a

cd "$SLURM_SUBMIT_DIR"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export JULIA_EXCLUSIVE=1

julia -t 4 prep.jl "$SLURM_ARRAY_TASK_ID"
```

`gf.slurm`:

```bash
#!/bin/bash
#SBATCH -J gf
#SBATCH --array=1-2
#SBATCH -N 1                   # lean follow-up job: 1 standard node
#SBATCH -n 1
#SBATCH -c 36
#SBATCH --mem=0
#SBATCH -t 7-00:00:00          # long: the hundreds of tdvp1 steps
#SBATCH -o output/gf_%A_%a

cd "$SLURM_SUBMIT_DIR"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export JULIA_EXCLUSIVE=1

julia -t 4 gf.jl "$SLURM_ARRAY_TASK_ID"
```

Submit the preparation job first, then the follow-up with a dependency so
it starts only after the preparation succeeds:

```bash
PREP=$(sbatch --parsable prep.slurm)
sbatch --dependency=afterok:$PREP gf.slurm
```

For small or medium systems the direct (non-split) multi-source method
`dcorrelator(gs, H, op, indices; ...)` is still available, with one
checkpoint file per source channel. Afterwards, transform the
real-space/time data to spectra with `fourier_kw` or `fourier_rw` (see
[Spectral Functions](tutorials/spectral_functions.md)).

## Guide

- [Getting Started](tutorials/getting_started.md): the package layout and a
  compact end-to-end workflow.
- [Ground State with DMRG](tutorials/dmrg.md): `dmrg1`/`dmrg2` with explicit
  `truncdims` schedules and the default algorithm constructors.
- [Dynamical Correlations](tutorials/dynamical_correlations.md):
  zero-temperature real-time correlators and CBE-TDVP1.
- [Spectral Functions](tutorials/spectral_functions.md): real-space/time to
  momentum-frequency transforms.
- [Finite Temperature](tutorials/finite_temperature.md): purification,
  imaginary-time preparation, and finite-temperature correlators.
- [The Finite Engine](tutorials/finite_engine.md): lazy/disk-backed
  environments and the threading layout behind the DMRG and TDVP drivers.
- [Momentum-Space Geometry](tutorials/momentum_space.md): Brillouin zones,
  allowed momentum cuts of (twisted) supercells, and plotting.

## API Reference

The API pages list exported models, operators, states, algorithms, observables,
and utility functions. Start with:

- [Algorithms](api/algorithms.md)
- [States](api/states.md)
- [Observables](api/observables.md)
- [Operators](api/operators.md)

## Compatibility Note

Before v1.0, minor versions may change APIs when the internal workflow improves.
Use the exported `my*` constructors (`myDMRG2`, `myDMRG1_CBE`, `myTDVP1`,
`myTDVP1_CBE`, `myTDVP2`) for the package's default algorithm configurations.

## Acknowledgments

This package builds on MPSKit.jl, TensorKit.jl, MPSKitModels.jl, and
QuantumLattices.jl. We thank the developers of those packages and the users who
have helped test the CBE and dynamical-correlation workflows.
