# DynamicalCorrelators.jl

*A convenient frontend for calculating dynamical correlation functions and related observables based on matrix-product states time evolution methods.*

See documents: https://Quantum-Many-Body.github.io/DynamicalCorrelators.jl

## Basic Workflow on HPC

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
real-space/time data to spectra with `fourier_kw` or `fourier_rw`.

## Installation

Please type `]` in the REPL to use the package mode, then type this command:

```julia
add DynamicalCorrelators
```

# DynamicalCorrelators.jl v0.15.0 Release Notes

## Highlights

v0.15.0 is a **workflow and analysis release** on top of the fast finite
engine introduced in v0.14.0. The largest new capability is the time-split
(meet-in-the-middle) zero-temperature dynamical correlator with a
distributed two-job workflow that separates the memory-heavy preparation
pass from the cheap long-time evolution. This release also adds penalized
excited-state DMRG, momentum-window MPS states, momentum-space geometry and
plotting helpers, linear prediction and Lehmann-representation
post-processing, and a rebuilt, lazy high-level API for static correlators
(**breaking**). Under the hood, the package moves to MPSKit 0.14 /
TensorKit 0.17.x, gains mid-sweep (update-level) DMRG checkpointing, and
the threaded effective-Hamiltonian paths were reworked around manual arena
allocators for much lower allocation pressure.

## Time-split dcorrelator and the distributed two-job workflow

- New time-split scheme `dcorrelator(gs, H, op; times, block, ...)`: both
  sides of the operator sandwich are evolved to `τ = t/2`, so every state
  is only evolved up to `tmax/2`, halving the peak entanglement and the
  accumulated truncation error compared to the direct method:

```julia
gf_greater = dcorrelator(gs, H, cp; times, block = :greater, ...)
gf_lesser  = dcorrelator(gs, H, cm; times, block = :lesser,  ...)
gfxt = gf_greater .+ gf_lesser
```

  The phase/conjugation conventions match the direct method exactly. The
  full `length(H) × length(H)` matrix of site pairs is computed at every
  time slice.
- `dcorrelator(...; mode = :distributed, workdir = ...)` evaluates every
  source state as an independent, checkpointed task on Distributed workers:
  a killed job resumes each state from its last checkpoint, and
  `charged_per_node` throttles how many memory-heavy charged-state
  constructions run on one host.
- `prepare_dcorrelator(gs, H, op; block, workdir, ...)` is the preparation
  pass: it computes the charged initial states and evolves them only
  through the two-site TDVP window — the memory high-water mark of the
  whole run. Submit it as a short, generously provisioned job, then submit
  a lean follow-up job calling `dcorrelator(...; mode = :distributed,
  workdir = <same>)`, which resumes every state from its checkpoint and
  continues with the cheap one-site TDVP. Both passes are idempotent and
  guarded by a shared `meta/config.jld2`.

## Excited-state DMRG

- New `excited_dmrg1!`/`excited_dmrg2!` drivers and the `excited_states`
  wrapper compute low-lying eigenstates in a fixed symmetry sector by the
  penalty method, with the local updates running on the fast finite engine:
  the Hamiltonian lives in a disk-backable `HalfFiniteEnvironments` and
  each penalty term is the exact rank-one operator built from plain two-leg
  overlap environments.
- After the `truncdims` schedule is exhausted, tail sweeps continue until
  `|ΔE| ≤ etol` on the *bare* energy, so an under-converged state is never
  passed on to the next penalized run. The optional `polish` pass re-sweeps
  every computed state against the final versions of all the others.

## MomentumMPS

- New `MomentumMPS` state type: momentum-window MPS (q-MPS) of Van Damme
  and Vanderstraeten, a thin wrapper around a collection of MPSKit
  `WindowMPS` objects, one per inequivalent site of the background unit
  cell, all sharing one Bloch momentum.
- Helpers: `qmps(gs::InfiniteMPS, O, q)` builds a momentum-window Bloch
  source at momentum `q` from a uniform ground state and a one-site
  operator; `translation_phase` and `check_momentum_mps` support and
  validate the window construction.

## Momentum-space geometry and plotting

- New geometry layer for momentum-space workflows: `BravaisCell2D`,
  `Supercell2D`, `TwistedTranslation`, and `MomentumGeometry`, with
  momentum-path types `MomentumCut`, `MomentumSegment`, and
  `MomentumPoint`.
- Query functions `brillouin_zone`, `allowed_cuts`, `bz_segments`,
  `allowed_points`, `is_allowed`, and `nearest_cut` give the allowed
  momenta of (twisted) finite supercells.
- `momentum_plot` visualizes geometries, Brillouin zones, and cuts through
  the new `DynamicalCorrelatorsMakieExt` CairoMakie extension (weak
  dependency, loaded on demand).
- A new momentum-space tutorial covers the geometry and plotting workflow.

## Lazy CorrelationPlan for static correlators (breaking)

- The old high-level correlator API (`AbstractCorrelation`,
  `PairCorrelation`, `TwoSiteCorrelation`, `OneSiteCorrelation`,
  `site_indices`, `pair_amplitude_indices`) is removed. The replacement is
  a lazy plan: channels register geometry only, and all tensor contractions
  happen in one shared, threaded evaluation:

```julia
plan = CorrelationPlan(gs)
add_one_site!(plan, :density; operator = n, lattice = lat, orbital = 1)
add_two_site!(plan, :spin; operators = (sp, sm), lattice = lat,
              orbital = 1, is = is, js = js)
add_pairing!(plan, :d_plus_id; operators = (sd, sl), lattice = lat,
             neighbors = Neighbors(1 => 1), orbitals = (1, 1),
             amplitude = f, is = is, js = js)
results = correlator(plan)   # NamedTuple keyed by channel name
```

- Raw expectation values are cached by operator object identity and MPS
  site indices — amplitudes are *not* part of the key, so different form
  factors (extended-s, d-wave, d+id, ...) over the same bonds share the
  same raw pair–pair values, and repeated evaluation or newly added
  channels never recompute cached work. `raw_call_counts(plan)` reports the
  raw kernel calls for diagnostics.
- Two-site and pairing channels return `is × js` matrices (the old
  paired-list convention is gone); one-site channels return a vector over
  the lattice sites. On-site pairing is a pairing channel with
  `Neighbors(0 => 0)` and shares the two-site raw cache. The distributed
  "np" mode was removed; evaluation is threaded on shared memory
  (`parallel` keyword).
- New `singlet`/`singlet_dagger` pair operators for fℤ₂ × U(1) × U(1)
  (bond and on-site, `side`/`filling` keywords), so pair–pair correlators
  no longer need charged-MPS overlaps for U(1) × U(1) problems.

## CBEExpand

- New `CBEExpand`: the direct CBE selection of this package as an MPSKit
  `changebond!` algorithm, usable anywhere MPSKit accepts `alg_expand`.
  `myDMRG1_CBE` and `myTDVP1_CBE` now configure `CBEExpand` (they used
  MPSKit's `OptimalExpand` in v0.14.0), and the DMRG drivers default to
  `D -> CBEExpand(; trunc = truncrank(ceil(Int, 0.1*D)))`.
- Falls back to an explicit matvec for states with multiple physical legs.

## Dependency upgrade

- MPSKit `0.14` (from a tracked main branch), TensorKit `0.17.x`,
  QuantumLattices `0.16`, MatrixAlgebraKit `0.6`, BlockTensorKit `0.3`,
  JLD2 `0.6`. New dependency: VectorInterface. New weak dependency:
  CairoMakie (for `momentum_plot`).

## Fast engine and threading internals

- `FastFiniteEnvironments` renamed to `HalfFiniteEnvironments`.
- The threaded Jordan-MPO effective-Hamiltonian paths now run on manual
  arena allocators (`BufferAllocator`) with staged-contraction caching and
  cost-ordered task lists, cutting allocation pressure and GC time in
  multithreaded DMRG/TDVP; Krylov vector operations are threaded as well.
- CBE contractions were factorized and the graph execution unified, with
  expand sub-timers read from the whole timer tree; TDVP gained per-stage
  diagnostics.
- `chargedMPS!` and `fast_approximate!` are exported for the fast
  variational compression path; `threaded_hamiltonian_threads` reports the
  configured Hamiltonian thread count.

## DMRG checkpointing

- DMRG drivers now write rolling atomic checkpoints by default, and support
  update-level (mid-sweep) checkpoint and resume: a killed sweep restarts
  from the last completed local update, with the update-level checkpoint
  placed next to the (possibly disk-backed) environments.

## Fourier and post-processing

- New `"GN"` broadening type: window-normalized Gaussian
  `exp(-η(t/tmax)²)`, whose frequency resolution keeps improving as the
  time window grows (typical `η ≈ 2 ~ 4`).
- `fourier_kw` and `fourier_rw` accept `symmetrize` (fold the two-sided
  time integral using `G(-t) = ±conj(G(t))`) and `kernel` (`:exp` or
  inversion-symmetric `:cos`) keywords.
- Linear prediction for time series: `LinearPrediction`,
  `LinearPredictionModel`, `fit_linear_prediction`, `extend_series`,
  `extrapolate_series`, `validation_error`, and `pole_analysis`.
- Lehmann-representation tools: `recommend_tmax`, `lehmann_eval`, and
  `lehmann_report`.
- `regroup_by_basis` partitions MPS sites by basis content (e.g.
  sublattices) for grouped Fourier transforms.

## API Changes

- Package version bumped from `0.14.0` to `0.15.0`.
- Removed exports: `AbstractCorrelation`, `PairCorrelation`,
  `TwoSiteCorrelation`, `OneSiteCorrelation`, `site_indices`,
  `pair_amplitude_indices`, `FastFiniteEnvironments`,
  `set_prefused_hamiltonian!`.
- New exports: `CorrelationPlan`, `add_one_site!`, `add_two_site!`,
  `add_pairing!`, `raw_call_counts`, `HalfFiniteEnvironments` (rename),
  `CBEExpand`, `excited_dmrg1!`/`excited_dmrg2!`, `excited_states`,
  `MomentumMPS`, `check_momentum_mps`, `translation_phase`, `qmps`,
  `BravaisCell2D`, `Supercell2D`, `TwistedTranslation`,
  `MomentumGeometry`, `MomentumCut`, `MomentumSegment`, `MomentumPoint`,
  `brillouin_zone`, `allowed_cuts`, `bz_segments`, `allowed_points`,
  `is_allowed`, `nearest_cut`, `momentum_plot`, `prepare_dcorrelator`,
  `chargedMPS!`, `fast_approximate!`, `threaded_hamiltonian_threads`,
  `regroup_by_basis`, `LinearPrediction`, `LinearPredictionModel`,
  `fit_linear_prediction`, `extend_series`, `extrapolate_series`,
  `validation_error`, `pole_analysis`, `recommend_tmax`, `lehmann_eval`,
  `lehmann_report`.
- `myDMRG1_CBE`/`myTDVP1_CBE` now return MPSKit algorithms configured with
  `CBEExpand` instead of `OptimalExpand`.

---

# DynamicalCorrelators.jl v0.14.0 Release Notes

## Highlights

v0.14.0 is a **migration and performance release** tracking the MPSKit.jl main
branch (with TensorKit v0.17.1). MPSKit main now natively supports Controlled
Bond Expansion through the `alg_expand` keyword of `DMRG` and `TDVP`
(`OptimalExpand`, `SketchedExpand`, `RandExpand`), so the hand-written CBE
overrides in this package have been removed in favor of the upstream
implementations. On top of them, v0.14.0 adds schedule-driven DMRG drivers
with explicit per-sweep bond-dimension control, thread-parallel sparse
Jordan-MPO effective Hamiltonians for finite systems. The finite-system DMRG and TDVP drivers share a new
lazy, optionally disk-backed environment manager (the fast finite engine)
that removes the dominant memory bottleneck of large-bond-dimension runs.

## Migration to MPSKit main

- MPSKit is tracked on the `main` branch (post-0.13.13, with the bond-expansion
  and prepared-derivative rework); TensorKit is pinned to `0.17.1` and
  BlockTensorKit to `0.3.16`. TimerOutputs accepts both `0.5.29` and `1.x`
  (QuantumLattices needs a compat that allows TimerOutputs `1.x`).
- Removed the custom rewrites that upstream now covers: `dmrg1_cbe!` /
  `dmrg1_cbe`, the `TDVP1_CBE` algorithm type, `idmrg2`, and the old
  multithreaded Hamiltonian overrides.
- `myDMRG1_CBE` and `myTDVP1_CBE` are now thin constructors around
  `DMRG(; alg_expand = OptimalExpand(...), trunc = ...)` and
  `TDVP(; alg_expand = ..., trunc = ...)`.
- Follow upstream renames: `SvdCut(; trscheme)` -> `SvdCut(; trunc)`,
  `DMRG2(; trscheme)` -> `DMRG2(; trunc)`, and `environments(ψ, H)` ->
  `environments(ψ, H, ψ)`.

## Schedule-driven DMRG drivers

- New `dmrg1!`/`dmrg1`, `dmrg2!`/`dmrg2`, and hybrid `dmrg_mix!`/`dmrg_mix`
  drivers run an explicit `truncdims` schedule: `truncdims[i]` is exactly the
  kept bond dimension after sweep `i`, and the number of sweeps is exactly
  `length(truncdims)`.
- They run on the fast finite engine by default and keep MPSKit's per-update
  structure, so one-site sweeps support
  `alg_expand = CBEExpand(...)` (this package) or MPSKit's
  `OptimalExpand(...)`/`SketchedExpand(...)`/`RandExpand(...)` as drop-in
  instances or a factory `D -> alg`; the default
  `D -> CBEExpand(; trunc = truncrank(ceil(Int, 0.1*D)))` overexpands each
  bond by 10% of the sweep target ahead of the eigensolve, assembled channel
  by channel without materializing the two-site effective Hamiltonian.
- Engine keywords shared by all drivers: `disk` (`false`, `true` for a
  `tempdir()` subdirectory, or a directory path) backs the environments by
  disk; `manual_gc` (default `false`), when enabled, runs a full collection
  and reports memory after every sweep; `envs` accepts an explicit environment manager — pass
  MPSKit's `environments(ψ, H, ψ)` to run the reference full-cache driver for
  cross-checks.
- `dmrg_mix` combines a robust two-site warmup at small D with cheap one-site
  CBE sweeps at large D, in a single run with continuous sweep numbering and
  checkpointing:

```julia
ψ, envs, E0 = dmrg_mix(ψ0, H, [64, 128, 256], [512, 1024, 1024];
                       alg_expand = D -> CBEExpand(; trunc = truncrank(ceil(Int, 0.3*D))))
# or equivalently with a switch point
ψ, envs, E0 = dmrg_mix(ψ0, H, [64, 128, 256, 512, 1024, 1024]; switch_D = 256)
```

- Per-move logging (`verbose = 2`), per-sweep summaries (`verbose = 1`),
  `TimerOutput` breakdowns, and JLD2 checkpoints via `filename`/`save_iters`
  are shared by all drivers.

## Threaded Jordan-MPO effective Hamiltonians

- For finite `MPOHamiltonian{<:JordanMPOTensor}` problems, the sparse
  continuing `A`/`AA` blocks of the one- and two-site effective Hamiltonians
  are evaluated as independent channels across Julia threads (per-task local
  accumulation, no BLAS oversubscription).
- This applies transparently to `dmrg1`/`dmrg2`/`dmrg_mix`, `find_groundstate`,
  and TDVP paths whenever `Threads.nthreads() > 1`, and falls back to MPSKit's
  prepared operators for single-threaded runs or Hamiltonians with at most one
  continuing channel.
- Toggle with `set_threaded_hamiltonian!(false)`.
- On the 4x6 Hubbard strip benchmark, the threaded drivers recover and exceed
  the previous custom implementation's performance (e.g. CBE-DMRG1 schedules
  run about 1.4x faster with `OptimalExpand` and about 2.8x faster with
  `SketchedExpand` than the default MPSKit path).

## Fast finite engine

- The finite DMRG drivers and the TDVP time evolution behind
  `dcorrelator`/`evolve_mps` share a lazy environment manager
  (`HalfFiniteEnvironments`): each environment is built on first query, freed
  right after its last use within a sweep, and invalidated automatically via
  object-identity tracking when local tensors change. Peak environment storage
  drops from `2(N+1)` cached tensors (MPSKit's `FiniteEnvironments`) to about
  `N+O(1)` — the difference that makes large-D SU(2) runs fit in memory.
- `disk = true` (or a directory path) serializes the environments to disk, one
  file per entry, deleting each file when its entry is freed. Sharing one
  `disk` root across distributed workers is safe (every environment array gets
  its own random subdirectory).
- When `Threads.nthreads() > 1`, environment construction and the truncated
  SVDs (DMRG/TDVP gauge steps, the TDVP2 split) are parallelized across Julia
  threads automatically, on top of the threaded effective Hamiltonians; BLAS
  stays pinned to one thread. `configure_finite_engine!()` sets and prints the
  recommended layout.
- `fast_timestep!` is the engine's in-place TDVP entry point for finite MPS;
  the state must be complex before its environments are constructed.
- The zero-temperature `chargedMPS(op, gs, site, alg)` compression runs the
  same variational sweep on the fast engine (`fast_approximate!` /
  `chargedMPS!`): mixed `⟨gs|O|ψ⟩` environments with the lazy `N+O(1)`
  storage model and the same `disk` keyword. The initial state is a zip-up
  warm start — one streaming MPO-MPS contraction sweep truncated with `alg`'s
  own gauge — rather than a random MPS, so the polish typically converges
  within a few sweeps. Override it with the `alg_zipup` keyword (e.g.
  `Zipup(; trunc = (truncrank(2D), truncrank(D)))`), or bring a custom
  initial state through `chargedMPS!`.

## Local eigensolvers

- DMRG local eigensolves use a fixed one-step `Lanczos` budget
  (`krylovdim = 16`, `tol = 1e-8`, `ModifiedGramSchmidt()`, eager early exit),
  the same conventional fixed-budget scheme as e.g. FiniteMPS.jl. (An earlier
  adaptive controller, MPSKit's `AdaptiveKrylov`, was tried as the default and
  reverted: its per-bond feedback misjudged the cold-start sweep of a
  bond-dimension ramp — cheap 3-dimensional solves followed by max-budget
  oscillation — while the converged-regime savings are already captured by
  eager early exit.)
- Pin a different budget with `krylovdim`, or pass `alg_eigsolve` explicitly:

```julia
alg = myDMRG1_CBE(; D = 1024)                        # fixed Lanczos (krylovdim = 16)
alg = myDMRG2(; trunc = truncrank(1024), krylovdim = 24)
```

- TDVP time integrators intentionally remain fixed `Lanczos`: the Krylov
  dimension needed by `exponentiate` is set by `dt` times the spectral width
  and does not drift during the evolution.

## Time-evolution memory usage

- `dcorrelator`/`evolve_mps` evolve in place with `fast_timestep!` on the fast
  finite engine: the state is promoted to complex once up front instead of
  copying the whole MPS at every step, environments follow the lazy `N+O(1)`
  memory model, and the `disk` keyword offloads them to disk at the largest
  bond dimensions.

## API Changes

- Removed exports: `TDVP1_CBE`, `dmrg1_cbe!`/`dmrg1_cbe`, `idmrg2`,
  `dmrg2_sweep!`, `myDMRG1CBE_eigsolve`.
- New exports: `dmrg1`/`dmrg1!`, `dmrg2`/`dmrg2!`, `dmrg_mix`/`dmrg_mix!`,
  `set_threaded_hamiltonian!`,
  `HalfFiniteEnvironments`, `fast_timestep!`, `configure_finite_engine!`,
  `free_left!`/`free_right!`, `env_memory_bytes`.
- `myTDVP1_CBE` keeps its name but now returns MPSKit's `TDVP` configured with
  `alg_expand = OptimalExpand(...)`; its `cbe_tol`/`project_error` keywords
  were removed (the upstream expansion is parameterized by `trunc` and
  `delta`).
- `myDMRG1`/`myDMRG2`/`myDMRG1_CBE` accept the `krylovdim` and
  `alg_eigsolve` keywords.

# DynamicalCorrelators.jl v0.13.0 Release Notes

## Highlights

v0.13.0 is a **Gorkov Green's function, superconducting-Hamiltonian, and
TDVP-CPT extension release**. Compared with v0.12.0, the main new capability is
that zero-temperature `dcorrelator` can now compute the four blocks of a Gorkov
Green's function for Hamiltonians with pairing terms, while reusing the same
expensive TDVP source evolution and adding only cheap projection sweeps.

This release also extends the TDVP-CPT workflow to imaginary-axis data and
grand-potential calculations, adds complex-frequency Fourier transforms,
improves timing/progress output for TDVP and CBE workflows, and expands the
fermionic operator/Hamiltonian helpers needed for SU(2)-symmetric pairing
models.

## Gorkov Green's Functions

- Added an `anomalous` keyword to the zero-temperature `dcorrelator` methods:

```julia
dcorrelator(gs, H, op, id; anomalous = other_op, ...)
dcorrelator(gs, H, op, indices; anomalous = other_op, ...)
```

- The default `anomalous = nothing` preserves the v0.12.0 normal Green's
  function behavior and return type.
- When `anomalous` is a `TensorMap`, `dcorrelator` returns a `NamedTuple`
  `(G11, G12, G21, G22)`, with each block having the same shape as the normal
  correlator output.
- For a creation source, e.g. `op = cre` and `anomalous = ann`, one TDVP
  evolution of `exp(-iHt)c_j^\dagger|0>` now produces all four Gorkov matrix
  positions associated with that source by evaluating both `sweep_dot(gs, cre,
  ket)` and `sweep_dot(gs, ann, ket)`.
- The complementary annihilation source, e.g. `op = ann` and
  `anomalous = cre`, similarly reuses the single
  `exp(-iHt)c_j|0>` evolution.
- This avoids splitting the Gorkov matrix into four independent TDVP
  calculations; the independent time evolutions remain only the two physical
  source states `c_j^\dagger|0>` and `c_j|0>`.
- Gorkov checkpoint files are stored separately as
  `gf_start=<t_start>_end=<t_end>_id=<id>_gorkov.jld2`, so normal and Gorkov
  runs do not collide on disk.
- Gorkov checkpoints include per-block completion checks for `G11`, `G12`,
  `G21`, and `G22`, and incomplete files are recomputed.

## Normal and Gorkov Documentation

- Rewrote the dynamical-correlations tutorial to state the actual formulas used
  by `dcorrelator`, including the ground-state phase factors
  `exp(+iE0*t)` and `exp(-iE0*t)`.
- Clarified source-channel terminology: ids `1:length(H)` are the greater
  contribution, while ids `length(H)+1:2length(H)` are the lesser contribution
  term used in the retarded combination.
- Added explicit normal Green's function and Gorkov Green's function formulas,
  including the block labels `G11`, `G12`, `G21`, and `G22`.
- Documented how to assemble a complete retarded Gorkov Green's function from
  the creation-source and annihilation-source runs.
- Updated the spectral-functions tutorial to use greater/lesser terminology
  instead of the older forward/conjugated wording.

## TDVP-CPT and Grand Potential

- Added a TDVP-CPT `GrandPotential` implementation for precomputed
  imaginary-axis cluster Green's functions stored in `cpt.gfrw`:

```julia
GrandPotential(cpt, bz, E0, iws; normal = nothing, weights = nothing)
GrandPotential(cpt, bz, E0; weights, normal = nothing)
```

- The implementation evaluates the discrete version of the VCA/CPT
  grand-potential integrand
  `log(abs(det(I - V(k) * G'(iω))))`.
- `GPcore` now uses `logabsdet` for a more stable determinant logarithm.
- If `weights` is omitted, trapezoidal weights are built from the supplied
  positive imaginary-frequency grid `iws`.
- Explicit quadrature weights can be supplied for nonuniform or externally
  generated imaginary-axis grids.
- Normal versus Gorkov CPT is inferred from the cluster Green's function matrix
  size: `N x N` is normal, while `2N x 2N` is Gorkov.
- The Gorkov grand-potential path includes the standard Nambu factor `1/2`.
- `singleParticleGreenFunction` now uses the same matrix-size inference, so the
  precomputed TDVP Green's function determines whether CPT runs in normal or
  Gorkov/Nambu space.

## Fourier Transforms

- Added complex-frequency and imaginary-axis transforms:

```julia
fourier_rz(gf_rt, ts, zs; broadentype = nothing, mthreads = Threads.nthreads(), ifsum = false)
fourier_riw(gf_rt, ts, iws; mu = 0, kwargs...)
```

- `fourier_rz` computes
  `G(z) = ∫ dt exp(i*z*t) G(t)`, which is useful for imaginary-axis CPT data
  with `z = mu + im*iω`.
- `fourier_riw` is a convenience wrapper for
  `fourier_rz(gf_rt, ts, mu .+ im .* iws; ...)`.
- `fourier_rw` now validates that the third axis of `gf_rt` matches the time
  grid.
- Fixed `fourier_rw` output sizing so non-square `(sink, source, time)` arrays
  preserve both the first and second dimensions instead of assuming a square
  matrix.

## Superconducting and SU(2)-Only Hamiltonian Support

- Added spin-SU(2)-only fermion Hamiltonian paths that use fermion parity
  instead of particle-number `U(1)`, enabling pairing terms that break particle
  number conservation.
- Added SU(2)-only directed hopping pieces:

```julia
cdagc(elt, SU2Irrep)
ccdag(elt, SU2Irrep)
```

- `hopping(elt, SU2Irrep)` is now built from these directed pieces.
- Added `FiniteMPO(::AbstractTensorMap)` construction through
  `add_single_util_leg` and `decompose_localmpo`, making local TensorMap
  operators easier to use as finite MPOs.
- Refactored `CustomLattice` Hubbard construction through `hubbard_terms`,
  with support for SU(2)-only pairing terms controlled by `se`.
- Added SU(2)-only bilayer two-band Hubbard construction and pairing fields
  `spmz` and `spmx` for interlayer pair terms on the `z` and `x` orbitals.
- Bilayer two-band parameter handling was moved into
  `hubbard_bilayer_2band_terms`, with zero-valued terms skipped through
  `iszero(...)` checks.

## Timing, Progress, and Profiling

- Added shared progress helpers for real-time evolution and dynamical
  correlator workflows, with compact elapsed-time formatting.
- `evolve_mps` and both zero-temperature and finite-temperature `dcorrelator`
  paths now use `TimerOutput` sections for setup, Hamiltonian construction,
  TDVP timesteps, `rho(t)` loading, and `sweep_dot`.
- `TDVP1_CBE` accepts an optional `timer` keyword in `timestep!`, allowing CBE
  expansion, AC/C Hamiltonian construction, and local time evolution to appear
  in the caller's timing report.
- Removed redundant timing wrappers in lower-level CBE calls so timing output is
  less noisy.

## API Changes

- Package version bumped from `0.12.0` to `0.13.0`.
- Chemical-potential keywords and examples now consistently use ASCII `mu`
  instead of Unicode `μ`:

```julia
hubbard(...; mu = 0.0)
conductivity(...; mu = 0)
```

- Updated docs, tests, and benchmark scripts to use the `mu` keyword.
- `S_plus`, `S_min`, and `heisenberg` for plain `SU2Irrep` spin operators now
  take the spin value as an explicit positional argument:

```julia
S_plus(elt, SU2Irrep, spin; side = :L)
S_min(elt, SU2Irrep, spin; side = :R)
heisenberg(elt, SU2Irrep, spin)
```

## Exports and Internal Updates

- Exported `GrandPotential`, `fourier_rz`, and `fourier_riw`.
- Imported `logabsdet` from `LinearAlgebra`.
- Kept `FiniteMPO` available through import from MPSKit while adding the local
  TensorMap-to-FiniteMPO convenience method.
- Updated tutorials and examples to reflect the current `mu` keyword and the
  Gorkov/greater/lesser correlator workflow.

---

# DynamicalCorrelators.jl v0.12.0 Release Notes

## Highlights

v0.12.0 is a **correlator-workflow and fermionic-string correctness release**.
Compared with v0.11.0, the largest user-facing change is that `dcorrelator`
now follows a single-operator workflow: pass one local operator and one source
id, or a collection of source ids. Source ids `1:length(H)` select the forward
channel, while `length(H)+1:2length(H)` select the conjugated channel.

Finite-temperature correlators also move to an explicit trajectory-on-disk
model. Instead of passing an in-memory `FiniteSuperMPS` to `dcorrelator`, first
save the real-time purification trajectory with `evolve_mps(...; save_id =
eachindex(times))`, then call `dcorrelator(rho_path, H, op, ids; ...)`.
During the correlator calculation, each worker loads only the current `rho(t)`
slice and evolves one charged source ket.

This release also tightens the handling of fermionic strings and charged
operator transfer matrices. String propagation now uses materialized
`TensorMap(BraidingTensor(...))` objects in the charged-MPO, finite-temperature
charged-MPS, `sweep_dot`, and multi-operator correlator paths, reducing
ambiguities in fermionic signs and braided transfer contractions.

## Dynamical Correlators

- Removed the old pair-operator `dcorrelator(..., (op₁, op₂); ...)` workflows.
  Use one operator at a time and select the conjugated channel through source
  ids `L+1:2L`, where `L = length(H)`.
- Zero-temperature `dcorrelator(gs, H, op, id; ...)` and
  `dcorrelator(gs, H, op, indices; ...)` now use the same single-operator
  channel convention.
- Finite-temperature methods now take the saved trajectory path as their first
  argument:

```julia
rho_path = "rho_realtime.jld2"
evolve_mps(H, times, rhoβ;
    filename = rho_path,
    save_id = eachindex(times),
)

gf = dcorrelator(rho_path, H, op, 1:length(H);
    times,
    beta = β,
)
```

- Finite-temperature correlators load `rho(t)` from keys of the form
  `"t=$(times[k])"` and compute all detector-site overlaps with `sweep_dot`
  instead of constructing every charged bra state explicitly.
- Completed JLD2 source files are still reused, and incomplete files are
  recomputed rather than silently accepted.
- The finite-temperature multi-source method evaluates source channels
  independently with `Distributed`, so each worker holds one charged ket and
  the current loaded `rho(t)` slice.

## `sweep_dot`

`sweep_dot` is now exported and documented for both zero-temperature and
finite-temperature workflows:

```julia
sweep_dot(gs, op, ket)
sweep_dot(rho, op, ket)
```

It computes all site overlaps in one sweep. Supported operator structures are
charge-carrying `(1,2)` operators using the `side = :L` convention and
charge-neutral `(1,1)` operators. Operators with the virtual leg on the
codomain side, i.e. `(2,1)`, are intentionally rejected in this sweep path;
construct the operator with `side = :L` for these correlator workflows.

## Fermionic Strings and Correlators

- `fZ(operator)` now returns a concrete
  `TensorMap(BraidingTensor(pspace, vspace))` instead of rebuilding the
  braiding contraction manually.
- `chargedMPO` documentation now matches the implemented string convention:
  `(1,2)` / `side=:L` operators place the string to the right of the source
  site, while `(2,1)` operators place it to the left.
- Finite-temperature `chargedMPS(op, rho, site)` now supports the `(1,2)`
  `side=:L` convention and `(1,1)` neutral operators, with explicit braiding
  tensors through the physical-ancilla legs.
- The static `correlator` contractions for charged two- and four-operator
  measurements were updated to propagate strings through intermediate sites
  with explicit transfer matrices and flip/isomorphism factors.
- Super-MPS transfer helpers were added for left/right environment propagation
  with and without charged strings.

## Hamiltonians and Operators

- Added and exported directed hopping operators:

```julia
cdagc(elt, SU2Irrep, U1Irrep; filling=(1, 1))
ccdag(elt, SU2Irrep, U1Irrep; filling=(1, 1))
```

- The SU(2) × U(1) Hubbard Hamiltonian now builds hopping terms from these two
  directed pieces, using conjugated amplitudes for the Hermitian partner terms.
- Zero-valued next-neighbor/interlayer hoppings and chemical-potential terms
  are skipped when constructing the Hamiltonian term list.

## Algorithm Defaults and API Changes

- Package version bumped from `0.11.0` to `0.12.0`.
- `myTDVP1` is now a constructor:

```julia
myTDVP1(; krylovdim = 32)
```

- `myTDVP2` now uses keyword arguments and has its own default truncation:

```julia
myTDVP2(; trunc = truncrank(4096), krylovdim = 16)
```

- The old `trscheme` keyword was removed from `evolve_mps` and `dcorrelator`
  call paths. Pass a complete TDVP2 object instead, for example:

```julia
dcorrelator(gs, H, op, 1:length(H);
    tdvp2 = myTDVP2(; trunc = truncrank(512)),
)
```

- `evolve_mps` defaults now construct algorithms as
  `tdvp1 = myTDVP1()` and
  `tdvp2 = myTDVP2(; trunc = truncerror(; rtol = 1e-3))`.
- `myDMRG1` now defaults to `tol = 1e-6`, matching the package's less aggressive
  default one-site DMRG tolerance.

## Internal Fixes and Documentation

- Corrected index ordering in the multithreaded Jordan-MPO `AC2_hamiltonian`
  action and in the direct CBE projector channel contractions.
- Simplified several CBE contractions by avoiding forced `@plansor opt=true`
  paths in numerically sensitive tensor contractions.
- Updated the documentation tutorials to show the current single-operator
  correlator interface, the finite-temperature `rho_path` workflow, and the
  exported `sweep_dot` API.

---

# DynamicalCorrelators.jl v0.11.0 Release Notes

## Highlights

v0.11.0 adds **CBE-TDVP1** for finite-MPS time evolution and extends the sparse
Jordan-MPO multithreading work from v0.10.0 to two-site effective Hamiltonians
and direct CBE projection. The new `TDVP1_CBE` algorithm follows MPSKit's
single-site finite-TDVP sweep structure, but expands each moving bond with the
direct Controlled Bond Expansion projector before the one-site TDVP update.

This release also changes the finite-temperature dynamical-correlator memory
model. Finite-temperature correlators now evolve the thermal state and active
source kets together, instead of precomputing and storing the full `rho(t)` and
all charged bra MPS objects for every time slice. Single-source correlator
methods are added for the one-operator zero-temperature and finite-temperature
cases, which avoids `SharedArray` and distributed scheduling when only one
source site is needed.

## CBE-TDVP1

The new `TDVP1_CBE` algorithm type is exported and has dedicated
`timestep` / `timestep!` methods:

```julia
TDVP1_CBE(; D=4096, cbe_tol=1e-10, delta=0.1, project_error=false)
```

The default constructor

```julia
myTDVP1_CBE(; D=4096, cbe_tol=1e-10, delta=0.1, project_error=false, krylovdim=32)
```

uses the same Lanczos time integrator style as the package's existing TDVP
defaults. The bond dimension is controlled by the `D` keyword: CBE temporarily
overexpands the active bond according to `delta`, runs the one-site TDVP update,
and truncates the shifted center back to the target dimension.

The implementation is designed to plug into existing time-evolution call sites
through the normal algorithm keyword path, for example by passing
`tdvp1 = myTDVP1_CBE(D=...)` in `dcorrelator` or `evolve_mps`.

## Multithreaded Two-Site and CBE Projectors

The sparse Jordan-MPO derivative-operator path now includes a multithreaded
two-site effective Hamiltonian:

```julia
AC2_hamiltonian(
    site::Int,
    below::FiniteMPS{<:MPSTensor},
    operator::MPOHamiltonian{<:JordanMPOTensor},
    above::FiniteMPS{<:MPSTensor},
    envs,
)
```

The continuing-continuing `AA` part is decomposed into nonzero sparse MPO
channels and evaluated across Julia threads, analogous to the v0.10.0
one-site `AC_hamiltonian` optimization.

The direct CBE projector used by `dmrg1_cbe!` also gets a specialized
Jordan-MPO implementation. It decomposes the projected two-site contraction
into `AA`, `CA`, `AB`, and `CB` channel groups and contracts the projected
intermediate directly as

```julia
NL' * channel * left_tensor * right_tail * NR'
```

without first materializing full two-site intermediates. The `AA`-dominant work
can therefore be evaluated in parallel while keeping the direct projector's
memory behavior.

## Dynamical Correlators

- Added zero-temperature single-source method
  `dcorrelator(gs, H, op, id; ...)`.
- Added finite-temperature single-source method
  `dcorrelator(rho, H, op, id; ...)`.
- These single-source methods write and resume the same `pro_k` checkpoint
  layout as the multi-source routines, but use ordinary arrays and a single
  local time-evolution path rather than `SharedArray` and distributed workers.
- `record_indices` handling for zero-temperature correlators is now validated
  explicitly and only evolves up to the last requested record.
- Incomplete JLD2 correlator files are detected and recomputed rather than
  silently accepted.
- Finite-temperature correlators no longer precompute the full thermal
  trajectory and all charged bra states. They keep only the current `rho_t`,
  the active source kets, and their environments during the time loop.
- `rho_path` is retained as a keyword for compatibility, but the v0.11.0
  finite-temperature implementation does not use cached `rho(t)` trajectory
  files.

## State Construction

The zero-temperature charged-state construction path now supports

```julia
chargedMPS(op, gs, site, alg)
```

which builds the charged state by running `approximate` from a random MPS in
the appropriate charged sector. The random-state helpers were updated with
`randFiniteMPS(elt, state, flux; side=:right)` and
`randFiniteMPS(elt, state, operator)` methods so the initial state can inherit
the bond-space profile of the reference MPS while carrying the operator's
boundary charge.

## API Changes

- Package version bumped from `0.10.0` to `0.11.0`.
- Default algorithm configuration names were renamed:
  - `DefaultDMRG` -> `myDMRG2()` for the default two-site DMRG constructor.
  - `DefaultDMRG2(tol, krylovdim)` -> `myDMRG2(; tol, krylovdim, ...)`.
  - `DefaultTDVP` -> `myTDVP1`.
  - `DefaultTDVP2(trscheme)` -> `myTDVP2(trscheme)`.
  - `DefaultDMRG1CBE_eigsolve` -> `myDMRG1CBE_eigsolve`.
- Added `myDMRG1()` for one-site DMRG and `myTDVP1_CBE()` for CBE-TDVP1.
- Exported `TDVP1_CBE` and `myTDVP1_CBE`.
- Replaced old `truncerr(...)` defaults with `truncerror(...)` in the affected
  time-evolution and correlator APIs.
- Documentation for default algorithms now points to the exported `my*`
  constructors.

## Validation

During development, the CBE-TDVP1 correlator path was compared against the
previous TDVP-based result and agreed at `atol=1e-4`, with relative difference
`norm(edc3 - edc4) / norm(edc3) = 2.6915052387074077e-6` in the tested case.

---

# DynamicalCorrelators.jl v0.10.0 Release Notes

## Highlights

v0.10.0 is a **performance release** for finite-MPS calculations with sparse
Jordan-MPO Hamiltonians. The main change is a new multithreaded one-site
effective Hamiltonian for MPSKit's `AC_hamiltonian` path. It keeps the original
Hamiltonian MPO structure unchanged, but rewrites the action of the sparse
`A` block in the local effective Hamiltonian so that only nonzero MPO channels
are contracted and these channels can be evaluated across Julia threads.

This release also updates the package version to `0.10.0`, adds
`BlockTensorKit`, and expands compatibility to newer `QuantumLattices` and
`TensorKit` versions.

## Multithreaded AC Hamiltonian

The threaded `AC_hamiltonian`/`AC2_hamiltonian` implementations specialize the
finite-MPS Jordan-MPO case:

```julia
AC_hamiltonian(
    site::Int,
    below::FiniteMPS{<:MPSTensor},
    operator::MPOHamiltonian{<:JordanMPOTensor},
    above::FiniteMPS{<:MPSTensor},
    envs,
)
```

Internally, the implementation enumerates every nonzero MPO path as one task,
keeps the staged contractions in a per-task scratch cache, sorts the task list
heavy-first, and executes it with `Threads.@threads :greedy` (or serially when
`set_threaded_hamiltonian!(1)` is set). Other MPS/operator combinations stay on
MPSKit's original dispatch path.

## Benchmarks

## DMRG/CBE Updates

- `dmrg1_cbe!` now defaults to `cbe_method = :direct`.
- The CBE eigsolve path now calls `AC_hamiltonian(...)` through the normal
  dispatch path, so finite-MPS Jordan-MPO calculations can use the new
  multithreaded AC Hamiltonian automatically.

## Compatibility

- Package version bumped from `0.9.1` to `0.10.0`.
- Added `BlockTensorKit = "0.3.13"` as a dependency.
- Expanded `QuantumLattices` compatibility to include `0.15`.
- Expanded `TensorKit` compatibility to include `0.17`.

---

# DynamicalCorrelators.jl v0.9.0 Release Notes

## Highlights

v0.9.0 is a **correctness-and-usability release** for CBE-DMRG1. The CBE
implementation is brought closer to the notation and sweep protocol of
Gleis, Li, von Delft, PRL 130, 246402 (2023): `delta` now denotes the paper's
working-space overexpansion factor, MPO preselection is controlled separately,
and projection-error reporting follows the CBE definition in the paper. This
release fixes a CBE preselection bug for non-Abelian symmetry-blocked MPOs,
where the MPO virtual dimension `w` was previously taken from the dense state
dimension instead of the multiplet count. It also adds finer timing diagnostics
for the four CBE shrewd-selection stages.

## CBE-DMRG1 fixes and improvements

### Paper-faithful dimension semantics

The meaning of `delta` is changed to match the "Sweeping" section of the paper:

```julia
D_work = ceil(Int, (1 + delta) * D_f)
D_tilde = D_work - D_i
```

The default remains `delta = 0.1`. Thus, each CBE update expands the current
bond from `D_i` toward the temporary working space `D_f(1+delta)`, performs the
1-site eigsolve, and then truncates back to `D_f` when shifting the isometry
center.

The previous preselection use of `delta` is moved to a new keyword:

```julia
preselect_factor = 1.0
```

The preselection rule is now

```julia
D′ = preselect_factor * D_f / w*
```

where `w*` is the MPO virtual-bond multiplet count. Use
`preselect_factor = :none` for the without-preselection variant `D′ = D_f`.
The default `preselect_factor = 1.0` corresponds to moderate preselection.

### Multiplet-aware MPO preselection

This release fixes an important correctness issue in CBE-DMRG1 with
non-Abelian symmetries. In previous versions, the MPO divisor `w` in
`D′ = D_f / w` was computed with `dim`, i.e. the dense state dimension of the
MPO virtual leg. For symmetry-blocked spaces such as `SU(2) × U(1)`, this is
not the quantity used in the paper: the algorithm should count MPO virtual
multiplets.

For symmetry-blocked Hamiltonians, the MPO divisor `w*` is now counted in
multiplets rather than dense state dimension. This avoids over-suppressing
`D′` for `SU(2) × U(1)` / fermion-parity block spaces, where the full dense
dimension can be much larger than the number of actual MPO virtual multiplets.
The old `dim`-based divisor could make preselection far too severe and lead to
incorrect or badly stalled CBE updates for non-Abelian symmetric calculations.

### Step-c numerical safety

CBE step (c) can optionally check whether the preselected complement remains
orthogonal to the kept MPS basis after repartitioning and SVD. The new
`safety` keyword defaults to `false` for speed. When enabled, nonorthogonal
preselected spaces are projected back into the complement and re-SVDed; bonds
that still fail the check are skipped and reported in the sweep summary.

The CBE selection SVDs now consistently use

```julia
truncrank(maxdim) & trunctol(atol = cbe_tol)
```

with `cbe_tol = 1e-10` by default. This tolerance controls numerical null
singular values in the CBE selection steps.

### Projection-error semantics

`ϵp` no longer reports the step-(b) preselection SVD truncation error. It now
follows the projection-error definition used in the CBE paper:

```julia
ϵp = |A_l A_{l+1} - A_l^ex A_{l+1}^ex|
```

when explicitly requested with `project_error = true`. By default,
`project_error = false`, and `ϵp` is reported as `NaN` to avoid the extra
two-site contraction cost.

### Canonical expansion and final truncation

Bond expansion now performs a final no-truncation orthogonalization after
`catdomain` / `catcodomain`. This keeps the enlarged basis canonical while preserving the
state before the eigsolve.

The post-eigsolve truncation uses the effective local target dimension, so the
kept bond is truncated back to the available `D_f`-limited space rather than
blindly asking for an impossible rank near physical or symmetry-sector
boundaries.

### Diagnostics

`TimerOutput` now breaks the CBE expansion into the four shrewd-selection
stages:

| Timer label | CBE stage |
|------|------|
| `CBE step a` | GETRORTH / mirrored GETLORTH |
| `CBE step b` | Preselection SVD |
| `CBE step c` | MPO-leg redirect and preselected basis |
| `CBE step d` | GETCORTH and final selection SVD |

The top-level `CBE expand`, `eigsolve`, and `SVD trunc` timers are retained.
When `safety = true`, skipped nonorthogonal bonds are printed as
`CBE nonorth skipped bonds`.

## API changes

- `delta` now means the paper's CBE overexpansion factor, default `0.1`.
- New keyword `preselect_factor`, default `1.0`, controls `D′`.
- New keyword `cbe_tol`, default `1e-10`, controls CBE selection SVD tolerance.
- New keyword `safety`, default `false`, enables step-c orthogonality checks.
- New keyword `project_error`, default `false`, enables explicit `ϵp`
  calculation.

## Internals

- Added multiplet counting through TensorKit `sectors`.
- Added `left_orth!` / `right_orth!` imports needed by the refined
  canonicalization path.
- Removed old debug-only CBE summary printing and replaced it with targeted
  safety diagnostics and timer sections.

---

# DynamicalCorrelators.jl v0.8.0 Release Notes

## Highlights

v0.8.0 is a **correctness-and-performance release** for CBE-DMRG1. The preliminary implementation in v0.7.0 (null-space projection via `AC2_hamiltonian`) is replaced by the full **four-step shrewd selection** procedure (preselection + final selection) of Fig. S-2 / Table I in [Gleis, Li, von Delft, PRL 130, 246402 (2023)](https://doi.org/10.1103/PhysRevLett.130.246402). The new implementation runs at strict 1-site cost `O(D³dw)`, matches reference results from FiniteMPS.jl v1.8.0 to within `2e-5` in ground-state energy, and is faster on the 4×6 Hubbard strip benchmark.

## CBE-DMRG1 rewrite

### Four-step shrewd selection (paper-faithful)

The core `_cbe_expand_l2r!` / `_cbe_expand_r2l!` orchestrators now call four explicit sub-steps corresponding to Fig. S-2 of the supplemental material:

| Step | L→R function | R→L mirror | Fig. S-2 role |
|------|------|---|---|
| (a) | `_cbe_getrorth_l2r` | `_cbe_getlorth_r2l` | GETRORTH / GETLORTH — project out kept space on the far side |
| (b) | `_cbe_getlorth_l2r` | `_cbe_getrorth_r2l` | Orthogonal-complement SVD, truncate `D → D′ = D_f/w` |
| (c) | `_cbe_get_Apr_l2r` | `_cbe_get_Bpr_r2l` | Redirect MPO leg, SVD to get preselected `Â^pr` with image dim `D̂ = D_f` |
| (d) | `_cbe_get_Atr_l2r` | `_cbe_get_Btr_r2l` | GETCORTH + final SVD → `Â^tr` with image dim `D̃` (closed MPO bond) |

Bond expansion `A^ex = A ⊕ Â^tr` and the zero-padded `C^{e.i.}` initial guess are factored out into `_cbe_bond_expand_l2r!` / `_cbe_bond_expand_r2l!`. All contractions are written with `@plansor opt=true` so the compiler picks a near-optimal contraction order at macro-expansion time.

### Two-regime D̃ schedule (ramp-up + maintain)

The expansion amount `D̃` is now chosen adaptively so the enlarged space **always** reaches the target `D_f·(1+δ)` before eigsolve:

```julia
D_tilde = max(round(Int, D_f * (1 + delta)) - D_current,
              round(Int, D_f * delta))
```

- **Ramp-up** (D_current ≪ D_f, e.g. right after `D_f` jumps from 64 to 256): the `D_target - D_current` branch dominates, filling the deficit in one bond update instead of creeping up by `δ·D_f` per sweep.
- **Maintain** (D_current saturated): the `δ·D_f` branch guarantees a persistent overexpansion cushion so Lanczos always has fresh directions to explore.

In v0.7.0 the expansion was fixed at `δ·D_f`, which under-expanded whenever `D_f` was raised mid-schedule and caused visibly slower convergence on the jump sweeps.

### Correct SVD absorption on the way back

After the enlarged-space eigsolve, the truncating SVD's `U` (L→R) or `Vᴴ` (R→L) is now absorbed back into `AL[pos]` / `AR[pos+1]` with `@plansor`, so the isometry's bond dimension truly returns to `D_f`. Previously the kept-side tensor retained the expanded bond, which caused `D` to drift well above the target across sweeps (observed: `D=244` instead of `D=64` on a `truncdims=[64]` run).

### SVD kernel swap: `svd_compact!` for untruncated SVDs

Step (a) uses `svd_compact!` (no trunc args) which is the TensorKit primitive for plain compact SVD; step (c) uses `svd_trunc!` with `truncrank(D_f) & trunctol(atol=1e-14)` to safely drop numerical-noise singular values that must vanish so `A†·Â^pr = 0`. `svd_compact!` is newly exported from the top-level module.

### API cleanup

- Positional signature unchanged: `dmrg1_cbe!(ψ, H, truncdims; kwargs...)`.
- Keyword `δ` renamed to `delta` (ASCII) to avoid unicode-only keyword kwargs.
- Removed the internal helper `_cbe_expand_dim(trscheme, current_D, δ)` — dimension arithmetic now lives inline in the two-regime formula above.

### Logging

Per-site sweep output is now right-aligned with width derived from `N` and `maximum(truncdims)`:

```
  SweepL2R: site  3 => site  4 | D =   64 | 22.Apr 2026 15:33
[3/6] CBE-DMRG1 sweep | duration: 5 seconds, 994 milliseconds
  E₀ = -10.4298628281 | D =  256 | ΔE = 5.594e-01 | ϵp = 8.220e-01 | err² = 0.000e+00
```

Summary line uses `@printf` for controlled precision (`.10f` on E₀, `.3e` on ΔE/ϵp/err²).

## Benchmark — 4×6 Hubbard strip, U=8, half-filling

`truncdims = [64, 64, 256, 256, 1024, 1024]`, `delta = 0.1`, 8 Julia threads (single-threaded BLAS).

### DynamicalCorrelators.jl v0.8.0

```
============================================================
  Hubbard model: 4×6 square lattice (FiniteStrip)
  N=24, t=1.0, U=8.0, half-filling
============================================================

── CBE + 1-site DMRG ──
CBE-DMRG1 Sweep Started: 22.Apr 2026 16:38
[1/6] CBE-DMRG1 sweep | duration: 3 seconds, 525 milliseconds
  E₀ = -9.5699390037 | D =   64 | ΔE = 5.885e+01 | ϵp = 1.625e+00 | err² = 2.685e-04
[2/6] CBE-DMRG1 sweep | duration: 4 seconds, 732 milliseconds
  E₀ = -10.0178418500 | D =   64 | ΔE = 4.479e-01 | ϵp = 1.177e+00 | err² = 1.650e-04
[3/6] CBE-DMRG1 sweep | duration: 8 seconds, 145 milliseconds
  E₀ = -10.4506952247 | D =  256 | ΔE = 4.329e-01 | ϵp = 8.046e-01 | err² = 6.199e-06
[4/6] CBE-DMRG1 sweep | duration: 9 seconds, 706 milliseconds
  E₀ = -10.5176702391 | D =  256 | ΔE = 6.698e-02 | ϵp = 4.142e-01 | err² = 2.582e-05
[5/6] CBE-DMRG1 sweep | duration: 58 seconds, 453 milliseconds
  E₀ = -10.5914846626 | D = 1024 | ΔE = 7.381e-02 | ϵp = 4.287e-01 | err² = 9.354e-07
[6/6] CBE-DMRG1 sweep | duration: 1 minute, 27 seconds, 582 milliseconds
  E₀ = -10.5962092176 | D = 1024 | ΔE = 4.725e-03 | ϵp = 1.792e-01 | err² = 2.316e-06
Ended: 22.Apr 2026 16:40 | total duration: 2 minutes, 52 seconds, 554 milliseconds
───────────────────────────────────────────────────────────────────────
                              Time                    Allocations
                     ───────────────────────   ────────────────────────
  Tot / % measured:        173s /  97.6%            219GiB /  99.7%

Section      ncalls     time    %tot     avg     alloc    %tot      avg
───────────────────────────────────────────────────────────────────────
eigsolve        276    98.2s   58.2%   356ms    148GiB   67.7%   549MiB
CBE expand      276    69.0s   40.8%   250ms   69.9GiB   32.0%   259MiB
SVD trunc       276    1.70s    1.0%  6.15ms    837MiB    0.4%  3.03MiB
───────────────────────────────────────────────────────────────────────
```

### FiniteMPS.jl v1.8.0 (reference)

```
Julia Version 1.12.5
FiniteMPS Version 1.8.0
Multi-threading Info:
 Julia: 8
 BLAS: 1
 action: 8
 mul: 1
 svd: 8
 eig: 8
BLAS Info:
 LBTConfig([ILP64] libopenblas64_.dylib)
============================================================
  Hubbard model: 4×6 square lattice (YCSqua)
  N=24, t=1.0, U=8.0, half-filling, U₁×U₁ symmetry
============================================================

── FiniteMPS.jl CBE-DMRG1 ──
[1/6] D=64,   E₀=-9.114597145457505,  time=28.49s
[2/6] D=64,   E₀=-10.099471418660887, time= 3.31s
[3/6] D=256,  E₀=-10.518607882664336, time= 4.65s
[4/6] D=256,  E₀=-10.524823146934540, time= 5.79s
[5/6] D=1024, E₀=-10.596158504571866, time=67.96s
[6/6] D=1024, E₀=-10.596219669830130, time=128.11s

Total time: 238.34s
```

## Breaking changes

- `dmrg1_cbe!` keyword `δ` → `delta`. Call sites that passed `δ=...` must be updated.
- Top-level `using TensorKit: ... svd_compact! ...` — downstream packages re-exporting from this module should add `svd_compact!` if they relied on the previous export list.
- No MPS / Hamiltonian / observable API changes.

## Internals

- Contractions inside CBE are all `@plansor opt=true`. `@macroexpand` confirms the auto-selected orders are sensible for the typical `D ≫ w > d` regime (Hubbard MPO block size `w = 7` on U(1)×U(1) Hubbard strip).
- `TimerOutput` sections now surface `CBE expand`, `eigsolve`, `SVD trunc` at the top level (finer sub-timings from v0.7.0's AC2-based path are gone because that path was removed).

## Miscellaneous

- `src/benchmark/` added with reproducible scripts for the CBE vs FiniteMPS / CBE vs DMRG2 comparisons used above.
- `Project.toml` bumped to `0.8.0`.
