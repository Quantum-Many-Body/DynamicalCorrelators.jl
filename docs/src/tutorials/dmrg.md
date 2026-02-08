# Ground State with DMRG

This tutorial covers how to find the ground state of a quantum many-body system using DMRG.

The core DMRG algorithms are provided by [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl).
DynamicalCorrelators.jl provides convenience wrappers (`dmrg2!`, `dmrg2_sweep!`, `idmrg2`)
that add progress logging, automatic file saving, and multi-sweep truncation management.

## Standard DMRG via MPSKit

The simplest approach uses MPSKit's `find_groundstate` directly:

```julia
using TensorKit, MPSKit, DynamicalCorrelators

filling = (1, 1)
H = hubbard(Float64, SU2Irrep, U1Irrep, FiniteChain(24); t=1, U=8, filling=filling)
st = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, 24; filling=filling)

gs, envs, delta = find_groundstate(st, H, DMRG2(trscheme=trunctol(1e-6)))
```

## Multi-sweep DMRG with `dmrg2!`

For production calculations, `dmrg2!` wraps MPSKit's DMRG2 with automatic progress logging
and JLD2 file saving after each sweep. It supports progressively increasing bond dimensions:

```julia
# Gradually increase bond dimension: 128 → 256 → 512 → 1024
truncdims = [128, 256, 512, 1024]

gs, envs, epsilon = dmrg2!(st, H, truncdims;
    filename="dmrg_results.jld2",
    verbose=true
)
```

This saves intermediate results to a JLD2 file after each sweep, allowing you to:
- Resume calculations if they are interrupted
- Monitor convergence by checking energy at each bond dimension
- Extract the optimal state at any truncation level

## Single-sweep DMRG with `dmrg2_sweep!`

For fine-grained control over individual sweeps:

```julia
# Initialize convergence tracker
epsilon = ones(Float64, length(st))

# Run sweeps manually
for iter in 1:10
    dmrg2_sweep!(iter, st, H, truncrank(1024), epsilon;
        filename="dmrg_sweep.jld2",
        verbose=true
    )
end
```

## Infinite DMRG with `idmrg2`

For infinite systems (translation-invariant), `idmrg2` wraps MPSKit's IDMRG/IDMRG2:

```julia
H_inf = hubbard(Float64, SU2Irrep, U1Irrep, InfiniteChain(2); t=1, U=8, filling=filling)
st_inf = randInfiniteMPS(ComplexF64, SU2Irrep, U1Irrep, 2; filling=filling)

gs_inf, envs_inf, delta_inf = idmrg2(st_inf, H_inf;
    filename="idmrg_results.jld2",
    verbose=true
)
```

## Tips for Production Runs

1. **Use `Float64` for DMRG**, then convert to `ComplexF64` for time evolution:
   ```julia
   gs_complex = complex(gs)
   ```

2. **Set MKL threads to 1** when using multi-process parallelism:
   ```julia
   using MKL
   BLAS.set_num_threads(1)
   ```

3. **Use OhMyThreads** for block-level parallelism in TensorKit:
   ```julia
   using OhMyThreads
   TensorKit.with_blockscheduler(DynamicScheduler()) do
       # DMRG code here
   end
   ```
