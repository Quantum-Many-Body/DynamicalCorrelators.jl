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
  through MPSKit's `DMRG(; alg_expand = OptimalExpand(...))`.
- `myTDVP1_CBE` configures CBE-assisted one-site TDVP through MPSKit's
  `TDVP(; alg_expand = OptimalExpand(...), trunc = ...)`. This lets single-site
  TDVP grow bonds through CBE while keeping the cheaper one-site time-evolution
  sweep.
- `dcorrelator` supports single-source and multi-source checkpointed
  real-time correlator calculations.
- finite-temperature correlators read a saved `rho(t)` trajectory one slice at
  a time and use sweep contractions against the active charged ket.

## Basic Workflow on HPC

A production zero-temperature run splits into two scripts: one DMRG job for
the ground state, then one job per operator/source batch for the dynamical
correlator. Both are submitted through the cluster scheduler (LSF `bsub`
below; translate to `sbatch` on SLURM), giving Julia all the cores of the
node — the finite engine parallelizes over Julia threads, so BLAS stays
pinned to one thread per task.

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

# blas_threads = 1: the engine parallelizes above BLAS (Hamiltonian channels,
# SVD blocks), so multi-threaded BLAS only adds contention.
# transformer_threads / manipulation_threads = nothing leaves TensorKit's own
# SU(2) recoupling thread pools unchanged; enable them only if profiling
# shows the fusion-tree transforms, not the contractions, dominate.
configure_finite_engine!(; blas_threads = 1,
                         transformer_threads = nothing,
                         manipulation_threads = nothing,
                         verbose = true)

# 6x6 square Hubbard with SU(2) spin x U(1) particle symmetry
coords = snake_2D([[1.0, 0.0], [0.0, 1.0]], vcat([[2,2,2,2,2,1,-2,-2,-2,-2,-2,1] for _ in 1:3]...)[1:end-1])
lattice = Lattice(coords...)
sq = Custom(lattice)
elt = Float64
t, u, filling = 1.0, 8.0, (1, 1)
H = hubbard(elt, SU2Irrep, U1Irrep, sq; t, U = u, filling)

# random initial state in the target charge sector; md is the (small)
# starting bond dimension — dmrg_mix grows it through the schedule
ψ = randFiniteMPS(elt, SU2Irrep, U1Irrep, length(H); md = 20, filling)

# dmrg_mix: cheap two-site sweeps first (truncdims_2site, they adapt the bond
# dimension to the entanglement structure), then polished by one-site sweeps
# with CBE bond expansion (truncdims_1site, one entry per sweep).
# trunc2 ramps 64 -> 8192; trunc1 then holds 8192 for 10 sweeps to converge.
trunc2 = [64, 1024, 4096, 8192]
trunc1 = [8192 for _ in 1:10]

ψ, envs, E0 = dmrg_mix!(ψ, H, trunc2, trunc1;
    # one-site sweeps cannot grow bonds by themselves: OptimalExpand adds up
    # to 10% new directions per bond ahead of each update. Increase the
    # fraction for frustrated/critical systems, decrease if runtime dominates.
    alg_expand = D -> OptimalExpand(; trunc = truncrank(ceil(Int, 0.1 * D))),
    # JLD2 checkpoint. save = true stores the final sweep; save = [2, 4, 6]
    # stores exactly those sweeps — restart a crashed job with
    # load(filename, "sweep_k_ψ") and feed it back as the initial state.
    filename = "/bbfs/fsa/username/jobname/hubbard_L=$(length(H))_t=$(t)_U=$(u).jld2",
    save = true,
    # disk: false keeps environments in RAM; true serializes them under
    # tempdir(); a string uses that directory. Essential at D ~ 4096+ on long
    # chains — point it at node-local scratch or a burst buffer (clusters
    # often mount one at /bbfs), not NFS.
    disk = "/bbfs/scratch/username/jobname",
    # GC after every local update + full collection per sweep (default true).
    # Keep it on at large D; turn off only for small tests.
    manual_gc = true,
    # 0 silent, 1 per-sweep summary, 2 also per-move lines
    verbose = 2,
)
```

Submit the whole node to a single multi-threaded Julia process (`-q` picks
the queue, `-m` the node, `-n 36` cores for `julia -t 36` threads; `-o`
names the log after the job ID):

```bash
bsub -q queue_name -m node_name -n 36 \
    -o "output/gs_%J" \
    julia -t 36 gs.jl
```

### Dynamical correlation (`gf.jl`)

The correlator of one operator for a batch of source sites runs on
Distributed workers — one charged ket per worker. Submit one job per batch
(here `i = parse(Int, ARGS[1])` selects the batch, e.g. from a job array).

```julia
using Pkg
Pkg.activate("$(ENV["HOME"])/envs/userenv/")   # the project environment with
                                               # DynamicalCorrelators & co.

using Distributed
i = parse(Int, ARGS[1])          # job-array index: which batch this job runs

using TensorKit
using MPSKit
using DynamicalCorrelators
using QuantumLattices
using JLD2: save, load

elt = Float64
t, u, filling = 1.0, 8.0, (1, 1)
L = 36

# reload the converged ground state from the checkpoint written by gs.jl
gs = load("/bbfs/fsa/username/jobname/hubbard_L=$(L)_t=$(t)_U=$(u).jld2", "sweep_14_ψ")

coords = snake_2D([[1.0, 0.0], [0.0, 1.0]], vcat([[2,2,2,2,2,1,-2,-2,-2,-2,-2,1] for _ in 1:3]...)[1:end-1])
lattice = Lattice(coords...)
H = hubbard(elt, SU2Irrep, U1Irrep, Custom(lattice); t, U = u, filling)

cp = e_plus(elt, SU2Irrep, U1Irrep; side = :L, filling)
cm = e_min(elt, SU2Irrep, U1Irrep; side = :L, filling)

# 18 worker processes, each with 4 Julia threads: worker count is limited by
# RAM (each worker holds a full charged ket + its environments), threads per
# worker drive the engine's channel/SVD parallelism inside each worker
addprocs(18; exeflags = `--threads=4`)

# workers start as bare Julia processes: they must activate the project
# environment themselves before loading packages
@everywhere begin
    using Pkg
    Pkg.activate("$(ENV["HOME"])/envs/userenv/")
    using TensorKit
    using MPSKit
    using DynamicalCorrelators
    using JLD2: save, load
end

# batch i of this job: which source sites and which operator
as = [1:18, 19:36, 37:54, 55:72]   # source-site batches (greater + lesser parts)
op = [cp, cp, cm, cm]

gf = dcorrelator(gs, H, op[i], as[i];
    # variational compression of the charged ket op|gs⟩ before evolving it;
    # match its trunc to the ground-state bond dimension you can afford
    approxalg = myDMRG2(; tol = 1e-6, maxiter = 50, trunc = truncrank(8192)),
    # first n = 3 time steps run the two-site tdvp2 (grows the charged ket's
    # bond dimension), later steps the cheaper one-site tdvp1
    tdvp2 = myTDVP2(; trunc = truncrank(8192)),
    tdvp1 = myTDVP1(),
    n = 3,
    times = 0:0.1:100,
    # per-channel checkpoints gf_*_id=$(id).jld2 land under gf_path; existing
    # files are overwritten, never reused
    gf_path = "/bbfs/fsa/username/jobname/gf_L=$(L)_U=$(u)_tmax=100/",
    # disk-backed environments, per-worker subdirectories (never collide)
    disk = "/bbfs/scratch/username",
)

save("/bbfs/fsa/username/jobname/gf_L=$(L)_U=$(u)_tmax=100/gf_xt_$(i).jld2", "gf", gf)
```

Submit one job per batch (72 cores = 18 workers × 4 threads; the main
process itself is nearly idle, so `julia` here needs no `-t` flag — the
workers get theirs from `addprocs(...; exeflags = `--threads=4`)`):

```bash
for i in {1..4}; do
    bsub -q queue_name -m node_name -n 72 \
        -o "output/output_${i}_%J" \
        julia gf.jl "$i"
done
```

Afterwards, transform the real-space/time data to spectra with `fourier_kw`
or `fourier_rw` (see [Spectral Functions](tutorials/spectral_functions.md)).

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
