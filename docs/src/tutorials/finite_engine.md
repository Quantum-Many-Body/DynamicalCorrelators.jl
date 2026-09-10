# The Finite Engine

All finite-system drivers in this package (`dmrg1`/`dmrg2`/`dmrg_mix`,
`dcorrelator`, `evolve_mps`) run on the package's *fast finite engine* by
default. The engine replaces MPSKit's fully-cached `FiniteEnvironments` with a
lazy, self-invalidating environment manager, parallelizes the serial stages of
a sweep across Julia threads, and can back the environments by disk. It is
automatic — the examples in the other tutorials already use it — but the
controls below matter at large bond dimensions and on clusters.

## Memory Model

MPSKit's `FiniteEnvironments` allocates both environment arrays up front and
keeps all `2(N+1)` environment tensors alive for the whole calculation. For
SU(2) Hamiltonians with wide MPO bonds this dominates the memory footprint at
large `D`.

`FastFiniteEnvironments` instead:

- builds each environment lazily on first query and frees it right after its
  last use within a sweep, so peak storage is about `N + O(1)` environment
  tensors instead of `2(N+1)`;
- tracks the exact `AL`/`AR` tensor object each environment was built from, so
  any local update (gauge, SVD split, CBE `changebond!`) automatically
  invalidates the dependent environments — no manual bookkeeping;
- optionally serializes environments to disk, one file per entry, deleting
  each file when the entry is freed.

## DMRG Driver Keywords

`dmrg1`/`dmrg2`/`dmrg_mix` accept three engine keywords:

- `disk` (default `false`): `true` serializes environments under a fresh
  random subdirectory of `tempdir()`; a string uses that directory as the root
  (created when missing). Point this at fast node-local or burst-buffer
  storage, not a shared network filesystem. Trade I/O for memory only when RAM
  is actually the bottleneck — run once with `disk = false` and watch the
  memory report first.
- `manual_gc` (default `true`): runs an incremental garbage collection after
  every local update and a full collection plus a memory report after every
  sweep. Set to `false` only for small systems.
- `envs`: pass an explicit environment manager to control its construction.
  Passing MPSKit's `environments(ψ, H, ψ)` selects the reference
  (full-cache) driver instead, which is useful for cross-checking the engine.

```julia
gs, envs, E0 = dmrg1(ψ0, H, truncdims;
    filename = "run.jld2",
    disk = "/scratch/myuser",   # or true for tempdir()
    verbose = 2,
)
```

## TDVP and Dynamical Correlators

`dcorrelator` and `evolve_mps` evolve the state in place with
`fast_timestep!` on the same engine, and accept the same `disk` keyword. The
charged state is promoted to a complex MPS once, up front; the TDVP1 gauge
(for truncating gauges such as CBE-TDVP) and the TDVP2 split use the
block-parallel SVD when Julia threads are available.

```julia
gf = dcorrelator(gs, H, op, 1:N;
    times,
    tdvp1 = myTDVP1_CBE(; D = 512),
    tdvp2 = myTDVP2(; trunc = truncrank(512)),
    disk = "/scratch/myuser",
)
```

To drive TDVP on the engine directly, build the environments yourself and call
`fast_timestep!` per step. The state must be complex before the environments
are constructed (their element types are fixed at construction):

```julia
ψ = complex(ψ)
envs = FastFiniteEnvironments(ψ, H; disk = true)
alg = myTDVP1()
for k in 2:length(times)
    fast_timestep!(ψ, H, times[k - 1], times[k] - times[k - 1], alg, envs)
end
```

With distributed workers (`@distributed` over source channels), sharing one
`disk` root directory between processes is safe: every environment array gets
its own random subdirectory underneath it. Note that a killed job leaves its
serialized entries behind; scratch filesystems with automatic cleanup are the
right target.

## Threading Model

The engine parallelizes *above* BLAS: environment transfer channels, effective
Hamiltonian channels, and SVD blocks are distributed over Julia threads, while
BLAS is pinned to one thread per call. Start Julia with several threads and
confirm the layout with:

```julia
configure_finite_engine!(; blas_threads = 1)
```

`transformer_threads`/`manipulation_threads` optionally enable TensorKit's
separate recoupling and index-manipulation thread pools on top.

When `Threads.nthreads() > 1`, three stages are parallelized automatically:

- the channel-split effective Hamiltonians (`AC`/`AC2` matvecs; toggle with
  `set_threaded_hamiltonian!(false)`, prefuse one-site channels with
  `set_prefused_hamiltonian!(true)`);
- environment construction (the transfer-matrix channel list);
- the block-parallel truncated SVD used by gauge steps and the TDVP2 split.

Single-threaded runs fall back to MPSKit's serial paths everywhere, so the
same scripts are valid with `-t 1`.
