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

### Dynamical correlation (the same 6×6 Hubbard model)

This section continues the ground-state example above with the **same 6×6
Hubbard Hamiltonian and checkpoint**. Because this model conserves particle
number, use the normal `:greater` and `:lesser` blocks, not the Gorkov block.
Each block has `2L = 72` independent states for `L = 36`; the two blocks use
separate workdirs and can be run independently.

Shared setup (`common_setup.jl`):

```julia
using Pkg
Pkg.activate("$(ENV["HOME"])/envs/userenv/")
using LinearAlgebra
using TensorKit, MPSKit, DynamicalCorrelators, QuantumLattices
using JLD2: load, save

elt = Float64
t, u, filling = 1.0, 8.0, (1, 1)
coords = snake_2D([[1.0, 0.0], [0.0, 1.0]],
    vcat([[2,2,2,2,2,1,-2,-2,-2,-2,-2,1] for _ in 1:3]...)[1:end-1])
lattice = Lattice(coords...)
H = hubbard(elt, SU2Irrep, U1Irrep, Custom(lattice);
    t=t, U=u, filling=filling)

tag = "hubbard_L=$(length(H))_t=$(t)_U=$(u)"
ckpt = "/bbfs/fsa/username/jobname/$(tag).jld2"
gs = load(ckpt, "ψ")
cp = e_plus(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)
cm = e_min(elt, SU2Irrep, U1Irrep; side=:L, filling=filling)

block = Symbol(get(ENV, "BLOCK", "greater"))
block in (:greater, :lesser) || error("BLOCK must be greater or lesser")
op = block === :greater ? cp : cm
times = 0:0.1:100
D = 8192
approxalg = myDMRG2(; tol=1e-6, maxiter=50, trunc=truncrank(D))
tdvp1 = myTDVP1()
tdvp2 = myTDVP2(; trunc=truncrank(D))
workdir = "/bbfs/scratch/username/jobname/gf_split_$(block)"
gf_path = "/bbfs/fsa/username/jobname/results_$(block)"
```

Each `state.jl` call handles one of the 72 states in the selected block:

```julia
include("common_setup.jl")

state = parse(Int, ARGS[1])
dcorrelator_state(gs, H, op;
    state, block, times, workdir,
    n=3, approxalg, tdvp1, tdvp2,
    disk="/bbfs/scratch/username/environments")
println("DONE state $state")
```

`overlap.jl` reads the same block configuration and writes one normal block:

```julia
include("common_setup.jl")

gf = dcorrelator_overlap(workdir;
    gsenergy=expectation_value(gs, H), gf_path, save=false, once=false,
    retention=2, poll_interval=30, overlap_block=16)
mkpath(gf_path)
save(joinpath(gf_path, "gf_xt_$(block).jld2"), "gf", gf, "times", times)
println("DONE")
```

For this large-`D` example, choose memory from measured per-state usage and the
partition's memory-per-core policy rather than copying a fixed value. The example
uses four cores/four Julia threads per state and eight cores/eight threads for
overlap. Always keep the CPU and Julia-thread counts equal:

```bash
#!/bin/bash
#SBATCH -J timesplit_state
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

The overlap job uses eight cores and eight Julia threads (`overlap.slurm`):

```bash
#!/bin/bash
#SBATCH -J timesplit_overlap
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

Submit each state once, selecting a currently suitable partition and node. Replace
the uppercase placeholders with values appropriate for your allocation:

```bash
d="$HOME/jobs/example_hubbard"
s=1
block=greater
partition=CPU_PARTITION
node=CPU_NODE
memory=MEMORY_REQUEST
sbatch --chdir="$d" -J "timesplit_state_${s}" -p "$partition" -w "$node" \
    -c 4 --mem="$memory" \
    --export="ALL,STATE=${s},BLOCK=${block},USE_MKL=1,OPENBLAS_NUM_THREADS=1" \
    "$d/state.slurm" "$d"
```

On AMD nodes, use `USE_MKL=0` instead. Submit the overlap independently with the
same `BLOCK` and an appropriate overlap memory request:

```bash
partition=GROUP_PARTITION
node=GROUP_NODE
overlap_memory=OVERLAP_MEMORY_REQUEST
sbatch --chdir="$d" -J timesplit_overlap -p "$partition" -w "$node" \
    --qos=QOS_NAME --mem="$overlap_memory" \
    --export="ALL,BLOCK=${block},USE_MKL=1,OPENBLAS_NUM_THREADS=1" \
    "$d/overlap.slurm" "$d"
```

Pass the directory as a quoted script argument as shown. Do **not** put a
`JOB_DIR` containing a comma in `--export`, because Slurm interprets commas as
separators. Absolute Julia driver paths and explicit `srun --chdir` prevent a
relative path from accidentally resolving under the home directory.

#### Reading state logs

Each state writes `output/state_<jobid>.out` and `.err`. A resumed four-thread
job can look like this:

```text
START state 1 | host cpu-node | Julia threads 4 | BLAS LBTConfig(...)
[state 1/72] bra site 1 on cpu-node | pid 12345 | threads 4
[state 1] resume from slice 22
[state 1] slice 23/1001 | t = 2.2 | maxD 8192 | wall 5m, 12s | GC 10.2% | RSS 24.6 GiB
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
[overlap] waiting for slice 14: 71/72 checkpoints
```

Once all 72 state checkpoints exist, the overlap is computed and saved:

```text
[overlap] slice 14/1001 | t = 1.3 | wall 18s
...
[overlap] complete: 1001/1001 slices
DONE
```

State jobs never wait for these messages or for the overlap job. Readiness is
determined from atomic checkpoint files, not log text. After each overlap is
saved, only old consumed checkpoints are removed; the latest two slices per
state remain available for resume. Therefore a finished run does not retain
every historical state file. It should have slice 1001 as every state's latest
checkpoint and all 1001 `meta/pro_NNNN.jld2` records. For this example,
`gf_xt_greater.jld2` and `gf_xt_lesser.jld2` each have shape
`(36, 36, 1001)`, and the retarded Green function is their sum.

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
