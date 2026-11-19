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
  scheme with independent one-state jobs and a standalone overlap job for
  production-size runs.
- Static correlators are computed with a lazy `CorrelationPlan`: channels
  register geometry only, and all raw expectation values are evaluated in one
  shared, threaded pass with automatic reuse between channels.
- finite-temperature correlators read a saved `rho(t)` trajectory one slice at
  a time and use sweep contractions against the active charged ket.

## Basic Workflow on HPC

A production zero-temperature run starts with a DMRG ground-state job. The
dynamical correlator then uses independent one-state jobs plus one standalone
overlap job, all submitted directly through SLURM (`sbatch`).

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

### Dynamical correlation (`state.jl` + `overlap.jl`)

Production runs use the time-split scheme with **one independent job per
state**. Each job computes its charged state when necessary, evolves through
the full time grid, and checkpoints every recorded slice. Jobs can land on any
suitable idle or mixed node; there is no master, preparation coordinator, or
dependency between state jobs.

Shared setup (`common_setup.jl`) defines a pairing-compatible `gs` and `H`, the
operators, and the common run parameters:

```julia
using Pkg
Pkg.activate("$(ENV["HOME"])/envs/userenv/")
using TensorKit, MPSKit, DynamicalCorrelators, QuantumLattices
using JLD2: load

state = parse(Int, ARGS[1])
gs_file = "/bbfs/fsa/username/jobname/gs.jld2"
gs = load(gs_file, "ψ")
# Construct the pairing-compatible H here, consistently with the ground state.
# H = ...

cp = e_plus(Float64, SU2Irrep; side=:L)
cm = e_min(Float64, SU2Irrep; side=:L)
times = 0:0.16:32
workdir = "/bbfs/scratch/username/jobname/gorkov/"
approxalg = myDMRG2(; tol=1e-6, maxiter=50, trunc=truncrank(8192))
tdvp1 = myTDVP1()
tdvp2 = myTDVP2(; trunc=truncrank(8192))
```

Each `state.jl` call handles only the requested state. For system size `L`, the
valid IDs are `1:4L`:

```julia
include("common_setup.jl")
dcorrelator_state(gs, H, cp;
    state, block=:gorkov, anomalous=cm, times, workdir,
    n=3, approxalg, tdvp1, tdvp2,
    disk="/bbfs/scratch/username/environments")
```

`overlap.jl` needs only the shared directory and ground-state energy:

```julia
using DynamicalCorrelators
using JLD2: load
workdir = "/bbfs/scratch/username/jobname/gorkov/"
E0 = load("/bbfs/fsa/username/jobname/gs.jld2", "E")
gf = dcorrelator_overlap(workdir;
    gsenergy=E0, gf_path="/bbfs/fsa/username/jobname/results",
    retention=2, poll_interval=30)
```

Example `state.slurm` for `L = 16` (64 independent array tasks, four cores each):

```bash
#!/bin/bash
#SBATCH -J gorkov_state
#SBATCH --array=1-64
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 7-00:00:00
#SBATCH -o output/state_%A_%a.log

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
export JULIA_EXCLUSIVE=1
julia -t 4 state.jl "$SLURM_ARRAY_TASK_ID"
```

`overlap.slurm` is an ordinary independent job:

```bash
#!/bin/bash
#SBATCH -J gorkov_overlap
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 7-00:00:00
#SBATCH -o output/overlap_%j.log

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
julia -t 4 overlap.jl
```

Submit the state jobs and the single overlap job directly; no Slurm dependency
is required. The overlap job waits for atomic checkpoint files, never for log
messages. After saving each overlap it removes only older, already-consumed
checkpoints and retains the latest two slices per state.

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
