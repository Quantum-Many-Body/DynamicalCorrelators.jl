# DynamicalCorrelators.jl

*A convenient frontend for calculating dynamical correlation functions and related observables based on matrix-product states time evolution methods.*

See documents: https://Quantum-Many-Body.github.io/DynamicalCorrelators.jl

## Installation

Please type `]` in the REPL to use the package mode, then type this command:

```julia
add DynamicalCorrelators
```

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
- `myTDVP` is now a constructor:

```julia
myTDVP(; krylovdim = 32)
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
  `tdvp1 = myTDVP()` and
  `tdvp2 = myTDVP2(; trunc = truncerror(; rtol = 1e-3))`.
- `myDMRG` now defaults to `tol = 1e-6`, matching the package's less aggressive
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
  - `DefaultTDVP` -> `myTDVP`.
  - `DefaultTDVP2(trscheme)` -> `myTDVP2(trscheme)`.
  - `DefaultDMRG1CBE_eigsolve` -> `myDMRG1CBE_eigsolve`.
- Added `myDMRG()` for one-site DMRG and `myTDVP1_CBE()` for CBE-TDVP1.
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

The new `JordanMPO_AC_Hamiltonian_Multithreading` implementation specializes
the finite-MPS Jordan-MPO case:

```julia
AC_hamiltonian(
    site::Int,
    below::FiniteMPS{<:MPSTensor},
    operator::MPOHamiltonian{<:JordanMPOTensor},
    above::FiniteMPS{<:MPSTensor},
    envs,
)
```

Compared with the MPSKit v0.13.10 implementation, the `D`, `I`, `E`, `C`, and
`B` terms are kept in the same contraction form. The improvement targets the
continuing `A` block, which can be very large as an MPO bond but contain only a
small number of nonzero block entries in symmetry-resolved Hamiltonians.

Internally, the implementation:

- collects valid sparse `A` channels with `BlockTensorKit.nonzero_pairs`;
- stores concrete `JordanMPO_AChannel_Multithreading` objects for type-stable
  local-Hamiltonian actions;
- evaluates the nonzero `A` channels with dynamic thread scheduling;
- leaves other MPS/operator combinations on MPSKit's original dispatch path.

This is especially useful for wide-MPO Hamiltonians where the MPO virtual bond
dimension is large but the local `A.data` block list is sparse.

## Benchmarks

On the local 16-site bilayer-Hubbard benchmark used during development, a
middle-site MPO block had an `A` right bond of 114 with only 23 nonzero block
entries. For this sparse-`A` case, the new AC action agrees numerically with
the original MPSKit action and is substantially faster:

| Operation | Implementation | Median time |
|------|------|------:|
| `Hac(x)` | MPSKit v0.13.10 AC Hamiltonian | `8.966 ms` |
| `Hac(x)` | `JordanMPO_AC_Hamiltonian_Multithreading` | `3.927 ms` |

The relative error in the tested `Hac(x)` result was at the level of
`~2e-18`. Effective-Hamiltonian construction itself remains essentially the
same cost as the original implementation:

| Operation | Implementation | Median time |
|------|------|------:|
| construct `AC_hamiltonian(...)` | MPSKit v0.13.10 | `2.046 ms` |
| construct `AC_hamiltonian(...)` | v0.10.0 multithreaded version | `2.051 ms` |

The speedup is therefore in repeated local-Hamiltonian actions, which is the
part repeatedly used by eigensolvers and time-evolution integrators.

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
