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

Production runs use the time-split scheme with **one independent Slurm job per
state**. Each job constructs its charged state when necessary, evolves through
the full time grid, and checkpoints every recorded slice. A separate overlap
job consumes completed slices. There is no preparation stage, master process,
Distributed worker pool, or dependency between jobs.

For a Gorkov calculation with system size `L`, there are `4L` state jobs:

| State IDs | State family | TDVP step |
|---|---|---|
| `1:L` | creation bras | backward |
| `L+1:2L` | annihilation bras | backward |
| `2L+1:3L` | creation kets | forward |
| `3L+1:4L` | annihilation kets | forward |

Thus `L = 16` requires 64 state jobs plus one overlap job. Normal
`:greater`/`:lesser` calculations have `2L` states instead.

Shared setup (`common_setup.jl`) defines a pairing-compatible `gs` and `H`, the
operators, and the common run parameters:

```julia
using Pkg
Pkg.activate("$(ENV["HOME"])/envs/userenv/")
using LinearAlgebra
using TensorKit, MPSKit, DynamicalCorrelators, QuantumLattices
using JLD2: load

gs_file = "/bbfs/fsa/username/jobname/gs.jld2"
gs = load(gs_file, "ψ")
# Construct the pairing-compatible H here, consistently with the ground state.
# H = ...

cp = e_plus(Float64, SU2Irrep; side=:L)
cm = e_min(Float64, SU2Irrep; side=:L)
times = 0:0.16:32
workdir = "/bbfs/scratch/username/jobname/gorkov/"
D = 2000
approxalg = myDMRG2(; tol=1e-6, maxiter=50, trunc=truncrank(D))
tdvp1 = myTDVP1()
tdvp2 = myTDVP2(; trunc=truncrank(D))
```

Each `state.jl` call handles only the requested state:

```julia
include("common_setup.jl")

state = parse(Int, ARGS[1])
println("START state $state | host ", Base.Libc.gethostname(),
        " | Julia threads ", Threads.nthreads(), " | BLAS ", BLAS.get_config())
dcorrelator_state(gs, H, cp;
    state, block=:gorkov, anomalous=cm, times, workdir,
    n=3, approxalg, tdvp1, tdvp2,
    disk="/bbfs/scratch/username/environments")
println("DONE state $state")
```

`overlap.jl` needs the shared directory and ground-state energy. It writes the
final complete Gorkov array after all 201 slices are available:

```julia
using DynamicalCorrelators
using JLD2: load, save

workdir = "/bbfs/scratch/username/jobname/gorkov/"
gf_path = "/bbfs/fsa/username/jobname/results"
E0 = load("/bbfs/fsa/username/jobname/gs.jld2", "E")
gf = dcorrelator_overlap(workdir;
    gsenergy=E0, gf_path, save=false, once=false,
    retention=2, poll_interval=30, overlap_block=16)
mkpath(gf_path)
save(joinpath(gf_path, "gf_xt_gorkov.jld2"), "gf", gf, "times", 0:0.16:32)
println("DONE")
```

The default state allocation is four cores, four Julia threads, and 12 GB. When
capacity is constrained, two cores, two Julia threads, and 9 GB can be used.
Always keep the CPU and Julia-thread counts equal. `state.slurm` obtains both
from Slurm so that command-line overrides cannot create a mismatch:

```bash
#!/bin/bash
#SBATCH -J gorkov_state
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 7-00:00:00
#SBATCH -o output/state_%j.out
#SBATCH -e output/state_%j.err

set -euo pipefail
: "${STATE:?STATE must be set}"
JOB_DIR=${1:?Job directory argument is required}
cd "$JOB_DIR"
mkdir -p output
export PATH="$HOME/.juliaup/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_DYNAMIC=FALSE
export KMP_BLOCKTIME=0
export JULIA_EXCLUSIVE=1
srun -n 1 -c "$SLURM_CPUS_PER_TASK" --chdir="$JOB_DIR" \
    julia -t "$SLURM_CPUS_PER_TASK" "$JOB_DIR/state.jl" "$STATE"
```

The overlap job uses eight cores, eight Julia threads, and 32 GB
(`overlap.slurm`):

```bash
#!/bin/bash
#SBATCH -J gorkov_overlap
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 7-00:00:00
#SBATCH -o output/overlap_%j.out
#SBATCH -e output/overlap_%j.err

set -euo pipefail
JOB_DIR=${1:?Job directory argument is required}
cd "$JOB_DIR"
mkdir -p output
export PATH="$HOME/.juliaup/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_DYNAMIC=FALSE
export KMP_BLOCKTIME=0
export JULIA_EXCLUSIVE=1
srun -n 1 -c 8 --chdir="$JOB_DIR" julia -t 8 "$JOB_DIR/overlap.jl"
```

Submit each state once, selecting a currently suitable partition and node. For
example, using four cores on `CPU_NODE`:

```bash
d="$HOME/jobs/example_gorkov"
s=1
sbatch --chdir="$d" -J "gorkov_s${s}" -p CPU_PARTITION -w CPU_NODE \
    -c 4 --mem=12G \
    --export="ALL,STATE=${s},USE_MKL=1,OPENBLAS_NUM_THREADS=1" \
    "$d/state.slurm" "$d"
```

On AMD CPU_PARTITION nodes, use `USE_MKL=0` instead. Submit the overlap independently;
for example, on the group node GROUP_NODE:

```bash
sbatch --chdir="$d" -J gorkov_overlap -p GROUP_PARTITION -w GROUP_NODE --qos=high \
    --export="ALL,USE_MKL=1,OPENBLAS_NUM_THREADS=1" \
    "$d/overlap.slurm" "$d"
```

Pass the directory as a quoted script argument as shown. Do **not** put a
`JOB_DIR` containing a comma in `--export`, because Slurm interprets commas as
separators. Absolute Julia driver paths and explicit `srun --chdir` prevent a
relative path from accidentally resolving under the home directory.

#### Reading state logs

Each state writes `output/state_<jobid>.out` and `.err`. A resumed two-thread
job can look like this:

```text
START state 1 | host cpu-node | Julia threads 2 | BLAS LBTConfig(...)
[state 1/64] bra site 1 on cpu-node | pid 12345 | threads 2
[state 1] resume from slice 22
[state 1] slice 23/201 | t = 3.52 | maxD 2000 | wall 3m, 44s, 17ms | GC 18.6% | RSS 5.89 GiB
...
DONE state 1
```

Here `resume from slice 22` confirms that the job did not restart from the
beginning. The next line reports the newly completed slice, physical time,
largest bond dimension, wall time for that step, garbage-collection fraction,
and memory use. During initial package precompilation or checkpoint loading,
stdout may remain buffered until the first new slice; inspect the `.err` file,
`sstat`, and checkpoint modification times before assuming the job is stalled.

#### Reading overlap logs

The overlap job writes `output/overlap_<jobid>.out`. While a slice is incomplete,
a periodic message such as this is normal:

```text
[overlap] waiting for slice 14: 61/64 checkpoints
```

Once all 64 state checkpoints exist, the overlap is computed and saved:

```text
[overlap] slice 14/201 | t = 2.08 | wall 18s
...
[overlap] complete: 201/201 slices
DONE
```

State jobs never wait for these messages or for the overlap job. Readiness is
determined from atomic checkpoint files, not log text. After each overlap is
saved, only old consumed checkpoints are removed; the latest two slices per
state remain available for resume. Therefore a finished run does not retain
every historical state file. It should have slice 201 as every state's latest
checkpoint, all 201 `meta/pro_NNNN.jld2` records, and the final finite
`gf_xt_gorkov.jld2` with shape `(32, 32, 201)` for this example.

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
