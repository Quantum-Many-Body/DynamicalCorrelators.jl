# Fast finite-system environment manager — Phase 1 of the finite-engine redesign.
#
# Memory model vs MPSKit's `FiniteEnvironments`:
#   * MPSKit's `initialize_environments` fills BOTH environment arrays with
#     `similar(boundary)` up front — i.e. 2(N+1) full-size environment tensors
#     are allocated before the first sweep — and never releases any of them.
#     For the 96-site SU(2) run with D = 8192 and MPO bond W = 113 that alone
#     is the ~225 GiB blow-up.
#   * Here environments are created lazily (`nothing` / no file until actually
#     built) and the sweep driver explicitly frees each environment right after
#     its last use (`free_left!` / `free_right!`), following FiniteMPS.jl's
#     Center/free! bookkeeping. Peak storage is ~N+O(1) environment tensors
#     instead of 2(N+1), with zero recomputation cost inside a sweep.
#
# Correctness model (identical to MPSKit): every stored `GLs[j+1]` records the
# exact `AL[j]` object it was built from (`ldependencies[j]`), and every query
# rebuilds the chain from the first stale entry. Because the local-update path
# always REPLACES tensor objects (`ψ.AC[pos] = ...`) instead of mutating them,
# object identity (`===`) detects staleness exactly — no manual invalidation
# calls are needed anywhere, including around CBE `changebond!` (a replaced
# `AL[pos]`/`AR[pos+1]` automatically makes the dependent environment stale).
#
# Storage conventions (identical to MPSKit, see finite_envs.jl):
#   GLs[i]    left environment contractible with site i  (i = 1:N+1, GLs[1] boundary)
#   GRs[j]    right environment, GRs[j] = TM(site j) * GRs[j+1]  (j = 1:N+1, GRs[N+1] boundary)
#   leftenv(env, i, ψ) == GLs[i];  rightenv(env, i, ψ) == GRs[i+1]
#
# With disk backing the two arrays are `SerializedElementArray`s (one file per
# entry, the same mechanism as FiniteMPS.jl's `disk = true`); freeing an entry
# deletes its file. Dependencies stay in memory (they are tiny references).
#
# The environment tensors remain ordinary MPSKit/BlockTensorKit objects, so the
# manager plugs into every MPSKit building block (`AC_hamiltonian`,
# `AC2_hamiltonian`, `calc_galerkin`, CBE `changebond!`, `expectation_value`,
# ...) through the `MPSKit.leftenv`/`MPSKit.rightenv` overloads below — the MPS
# and MPO themselves are untouched MPSKit objects.

"""
    HalfFiniteEnvironments(ψ::FiniteMPS, H; disk = false,
        threaded_transfer = Threads.nthreads() > 1)

Memory-frugal environment manager for finite MPSKit calculations.

# Keyword arguments
- `disk`: `false` (default) keeps environments in RAM; `true` serializes them
  under a fresh random subdirectory of `tempdir()`; a string uses that
  directory as the root (created when missing; each array gets its own random
  subdirectory underneath, so concurrent runs never collide). Disk backing
  trades I/O for memory at the largest bond dimensions; freed entries are
  deleted. Point this at fast node-local or burst-buffer storage (e.g. via
  `TMPDIR` or an explicit string like `"/bbfs/scratch/<user>"`), not at a
  shared network filesystem.
- `threaded_transfer`: build each environment from the Jordan-MPO channel list
  contracted across Julia threads (per-task local accumulation + locked merge,
  the same pattern as the threaded Hamiltonian). Defaults to
  `Threads.nthreads() > 1` — it parallelizes a stage that is otherwise serial,
  so there is nothing to conflict with; the keyword remains only as an escape
  hatch for verification/debugging. NOTE: the assembled per-block
  `BlockTensorMap` relies on `BlockTensorKit` internals (`SumSpace` leg
  assembly) — validate with `src/draft/verify_finiteengine.jl` before relying
  on it.

The boundary environments are constructed exactly as in
`MPSKit.environments(ψ, H, ψ)`, and interior environments are built on first
query, so `HalfFiniteEnvironments(ψ, H)` is a drop-in replacement for
`environments(ψ, H, ψ)` in the DMRG drivers.
"""
mutable struct HalfFiniteEnvironments{VL <: AbstractVector, VR <: AbstractVector, TA, O, A}
    GLs::VL                  # length N+1; GLs[i] contractible with site i
    GRs::VR                  # length N+1; GRs[j] right env of site j-1... see above
    ldependencies::Vector{Union{Nothing, TA}}  # ldependencies[j] = AL[j] object GLs[j+1] was built from
    rdependencies::Vector{Union{Nothing, TA}}  # rdependencies[j] = AR[j] object GRs[j] was built from
    operator::O
    # `nothing` for Hamiltonian environments (above === below, the DMRG/TDVP
    # case); the fixed bra-side MPS for variational-compression environments
    # `HalfFiniteEnvironments(ψ₀, O, ϕ)` (MPSKit's `environments(below, O, above)`).
    # The bra-side state never changes during a sweep, so only the queried
    # (below) state's tensors participate in the === dependency tracking.
    above::A
    threaded_transfer::Bool
end

function HalfFiniteEnvironments(
        ψ::FiniteMPS, H;
        disk::Union{Bool, AbstractString} = false,
        threaded_transfer::Bool = Threads.nthreads() > 1
    )
    N = length(ψ)
    N >= 2 || throw(ArgumentError("HalfFiniteEnvironments needs at least 2 sites"))
    S = site_type(ψ)

    # boundaries, exactly as MPSKit's `environments(below, operator, below)`
    Vl = left_virtualspace(ψ, 1)
    GL1 = isomorphism(storagetype(S), Vl ⊗ left_virtualspace(H, 1)' ← Vl)
    Vr = right_virtualspace(ψ, N)
    GRN = isomorphism(storagetype(S), Vr ⊗ right_virtualspace(H, N) ← Vr)

    TL = Union{Nothing, typeof(GL1)}
    TR = Union{Nothing, typeof(GRN)}
    if disk !== false
        # SerializedElementArray mkpaths the root and gives each array its own
        # random subdirectory underneath it
        path = disk === true ? tempdir() : String(disk)
        @info "HalfFiniteEnvironments: serializing environments under $path"
        GLs = serialize_disk(TL[i == 1 ? GL1 : nothing for i in 1:(N + 1)]; path)
        GRs = serialize_disk(TR[i == N + 1 ? GRN : nothing for i in 1:(N + 1)]; path)
    else
        GLs = TL[i == 1 ? GL1 : nothing for i in 1:(N + 1)]
        GRs = TR[i == N + 1 ? GRN : nothing for i in 1:(N + 1)]
    end
    TA = nonmissingtype(eltype(ψ.ALs))   # site-tensor type stored as dependencies
    ldependencies = Vector{Union{Nothing, TA}}(nothing, N)
    rdependencies = Vector{Union{Nothing, TA}}(nothing, N)
    return HalfFiniteEnvironments(
        GLs, GRs, ldependencies, rdependencies, H, nothing, threaded_transfer
    )
end

"""
    HalfFiniteEnvironments(below::FiniteMPS, O, above::AbstractFiniteMPS; disk = false)

Mixed (two-state) environment manager for variational compression
(MPSKit's `environments(below, operator, above)`): the environments of the
overlap ⟨above|O|below⟩, where `below` is the state being optimized (its
tensors are replaced during the sweep and drive the === staleness tracking)
and `above` is the fixed target state.

The boundary environments are constructed exactly as in MPSKit:
`GLs[1] = Vl_below ⊗ Vl_O′ ← Vl_above` and
`GRs[N+1] = Vr_above ⊗ Vr_O ← Vr_below`. Used by [`chargedMPS!`](@ref) /
`fast_approximate!` with `O = chargedMPO(op, site, N)` and `above = gs`.

`threaded_transfer` is not offered here: the channel-split threaded transfer
is a Jordan-MPO-Hamiltonian optimization, while the compression operator is a
plain (width-1/2) `FiniteMPO` whose transfer is a single small contraction.
"""
function HalfFiniteEnvironments(
        below::FiniteMPS, O, above::AbstractFiniteMPS;
        disk::Union{Bool, AbstractString} = false
    )
    N = length(below)
    N >= 2 || throw(ArgumentError("HalfFiniteEnvironments needs at least 2 sites"))
    length(above) == N || throw(DimensionMismatch(
        "below and above must have the same length (got $N and $(length(above)))"
    ))
    S = site_type(below)

    # boundaries, exactly as MPSKit's `environments(below, operator, above)`
    GL1 = isomorphism(
        storagetype(S),
        left_virtualspace(below, 1) ⊗ left_virtualspace(O, 1)' ← left_virtualspace(above, 1)
    )
    GRN = isomorphism(
        storagetype(S),
        right_virtualspace(above, N) ⊗ right_virtualspace(O, N) ← right_virtualspace(below, N)
    )

    TL = Union{Nothing, typeof(GL1)}
    TR = Union{Nothing, typeof(GRN)}
    if disk !== false
        path = disk === true ? tempdir() : String(disk)
        @info "HalfFiniteEnvironments: serializing environments under $path"
        GLs = serialize_disk(TL[i == 1 ? GL1 : nothing for i in 1:(N + 1)]; path)
        GRs = serialize_disk(TR[i == N + 1 ? GRN : nothing for i in 1:(N + 1)]; path)
    else
        GLs = TL[i == 1 ? GL1 : nothing for i in 1:(N + 1)]
        GRs = TR[i == N + 1 ? GRN : nothing for i in 1:(N + 1)]
    end
    TA = nonmissingtype(eltype(below.ALs))
    ldependencies = Vector{Union{Nothing, TA}}(nothing, N)
    rdependencies = Vector{Union{Nothing, TA}}(nothing, N)
    return HalfFiniteEnvironments(GLs, GRs, ldependencies, rdependencies, O, above, false)
end

length(env::HalfFiniteEnvironments) = length(env.GLs) - 1

"""
    env_memory_bytes(env::HalfFiniteEnvironments) -> Int

Approximate number of bytes currently held by the stored environments.
For disk-backed managers this only counts the (tiny) in-memory structures;
use the process RSS for the real footprint.
"""
env_memory_bytes(env::HalfFiniteEnvironments) =
    Base.summarysize(env.GLs) + Base.summarysize(env.GRs)

# ---------------------------------------------------------------------------
# storage primitives (memory vs disk)
# ---------------------------------------------------------------------------

_isstored(v::AbstractVector, i::Int) = v[i] !== nothing
_isstored(v::SerializedElementArray, i::Int) =
    isfile(filename(v, i))

_free_entry!(v::AbstractVector, i::Int) = (v[i] = nothing; nothing)
function _free_entry!(v::SerializedElementArray, i::Int)
    # same mechanism as FiniteMPS.jl's `cleanup!`: remove the serialized file
    rm(filename(v, i); force = true)
    return nothing
end

"""
    free_left!(env::HalfFiniteEnvironments, i::Int)

Free the left environment `GLs[i]` (the one contractible with site `i`). The
boundary `GLs[1]` cannot be freed. Freed environments are rebuilt lazily on
the next query; only call this for environments that will not be needed again
in the current sweep direction.
"""
function free_left!(env::HalfFiniteEnvironments, i::Int)
    i == 1 && throw(ArgumentError("the boundary left environment GLs[1] cannot be freed"))
    _free_entry!(env.GLs, i)
    env.ldependencies[i - 1] = nothing
    return env
end

"""
    free_right!(env::HalfFiniteEnvironments, j::Int)

Free the right environment `GRs[j]`. The boundary `GRs[N+1]` cannot be freed.
See [`free_left!`](@ref) for the usage contract.
"""
function free_right!(env::HalfFiniteEnvironments, j::Int)
    j == length(env.GRs) &&
        throw(ArgumentError("the boundary right environment GRs[N+1] cannot be freed"))
    _free_entry!(env.GRs, j)
    env.rdependencies[j] = nothing
    return env
end

# ---------------------------------------------------------------------------
# environment construction (single site transfer)
# ---------------------------------------------------------------------------

# Default path: literally MPSKit's expressions from finite_envs.jl — the full
# Jordan-MPO transfer through BlockTensorKit, bit-identical numerics to MPSKit.
# `state` is the queried (below) state; when `env.above` is set (mixed
# compression environments) the bra side of the transfer comes from the fixed
# above state instead (MPSKit's `leftenv`: `TransferMatrix(above, O, below)`).
# The threaded branch is never reached for mixed environments (their
# constructor fixes threaded_transfer = false).
function _pushright(env::HalfFiniteEnvironments, j::Int, state::AbstractFiniteMPS)
    A = state.AL[j]
    O = env.operator[j]
    if env.threaded_transfer && Threads.nthreads() > 1
        return _transfer_left_threaded(env.GLs[j], O, A)
    end
    Aa = env.above === nothing ? A : env.above.AL[j]
    return env.GLs[j] * TransferMatrix(Aa, O, A)
end

function _pushleft(env::HalfFiniteEnvironments, j::Int, state::AbstractFiniteMPS)
    A = state.AR[j]
    O = env.operator[j]
    if env.threaded_transfer && Threads.nthreads() > 1
        return _transfer_right_threaded(env.GRs[j + 1], O, A)
    end
    Aa = env.above === nothing ? A : env.above.AR[j]
    return TransferMatrix(Aa, O, A) * env.GRs[j + 1]
end

# ---------------------------------------------------------------------------
# MPSKit adapter: lazy queries with === staleness detection
# ---------------------------------------------------------------------------

function leftenv(env::HalfFiniteEnvironments, ind::Int, state::AbstractFiniteMPS)
    GLs = env.GLs
    a = findfirst(j -> !_isstored(GLs, j + 1) || !(state.AL[j] === env.ldependencies[j]), 1:(ind - 1))
    if !isnothing(a)
        for j in a:(ind - 1)
            GLs[j + 1] = _pushright(env, j, state)
            env.ldependencies[j] = state.AL[j]
        end
    end
    return GLs[ind]
end

function rightenv(env::HalfFiniteEnvironments, ind::Int, state::AbstractFiniteMPS)
    GRs = env.GRs
    N = length(state)
    a = findfirst(j -> !_isstored(GRs, j) || !(state.AR[j] === env.rdependencies[j]), N:-1:(ind + 1))
    if !isnothing(a)
        a = N - a + 1
        for j in a:-1:(ind + 1)
            GRs[j] = _pushleft(env, j, state)
            env.rdependencies[j] = state.AR[j]
        end
    end
    return GRs[ind + 1]
end

# ---------------------------------------------------------------------------
# channel-split threaded transfer (opt-in; VERIFY against MPSKit envs)
# ---------------------------------------------------------------------------
#
# BEFORE touching any contraction below, read
# docs/tensorkit-plansor-conventions.md (@plansor planar leg-order rules) and
# confirm the index scheme with the user first.
#
# The Jordan-MPO transfer decomposes exactly over the nonzero channels of W
# (linearity of the contraction):
#   GL′[..., r, ...] = Σ_{(I, Wij): I[4] == r} transfer_left(GL[..., I[1], ...], Wij, A, A)
# where `nonzero_pairs(W)` covers the full JordanMPOTensor (A/B/C/D blocks plus
# the materialized corner identities). Each channel is a dense contraction;
# channels are distributed over Julia threads with per-task local accumulation
# and a single locked merge per task — the same pattern as `_apply_channels`
# in algorithms/hamiltonian_threaded.jl.

# Work-stealing over channels; `f(pair) -> (slot, contribution)`; returns a
# vector of per-slot accumulated tensors (`nothing` where no channel landed).
# The first channel is peeled off and run eagerly so the accumulators get the
# concrete element type `Union{Nothing, typeof(y)}` instead of `Any`.
function _threaded_channel_accumulate(f, pairs, nslots::Int)
    isempty(pairs) && throw(ArgumentError("no channels to accumulate"))
    r0, y0 = f(first(pairs))
    T = typeof(y0)
    newacc() = Vector{Union{Nothing, T}}(nothing, nslots)
    if Threads.nthreads() == 1 || length(pairs) == 1
        acc = newacc()
        acc[r0] = y0
        for i in 2:length(pairs)
            r, y = f(pairs[i])
            acc[r] = acc[r] === nothing ? y : acc[r] + y
        end
        return acc
    end
    total = newacc()
    total[r0] = y0
    idx = Threads.Atomic{Int}(2)   # channel 1 already done above
    lk = ReentrantLock()
    nt = min(Threads.nthreads(), length(pairs) - 1)
    Threads.@sync for _ in 1:nt
        Threads.@spawn begin
            local_acc = newacc()
            while true
                i = Threads.atomic_add!(idx, 1)
                i > length(pairs) && break
                r, y = f(pairs[i])
                local_acc[r] = local_acc[r] === nothing ? y : local_acc[r] + y
            end
            lock(lk) do
                for r in 1:nslots
                    local_acc[r] === nothing && continue
                    total[r] = total[r] === nothing ? local_acc[r] : total[r] + local_acc[r]
                end
            end
        end
    end
    return total
end

# The summand spaces of a (possibly trivial) SumSpace.
_summands(V::SumSpace) = V.spaces
_summands(V) = (V,)

# Assemble per-block results into the rank-3 BlockTensorMap environment.
# `Vmid` is the middle-leg SumSpace of the output; each block has space
# `(Dcod ⊗ summand) ← Ddom` with ordinary (non-sum) virtual legs.
# VERIFY: the `ProductSumSpace`/`←` assembly relies on BlockTensorKit space
# promotion; a mismatch shows up as an error or wrong norm in verification.
function _assemble_env(blocks::AbstractVector{<:Union{Nothing, TT}}, Vmid) where {TT <: AbstractTensorMap}
    summands = _summands(Vmid)
    nb = length(summands)
    length(blocks) == nb || throw(DimensionMismatch(
        "channel results ($(length(blocks))) do not match middle-leg summands ($nb)"
    ))
    i1 = findfirst(b -> b !== nothing, blocks)
    i1 === nothing && throw(ArgumentError("no channel contributed to the environment"))
    y1 = blocks[i1]
    # NOTE: use codomain/domain, NOT space(y1, 1)/space(y1, 3) — space() returns
    # the DUAL of domain legs, which would assemble an env whose outer domain
    # leg carries a spurious dual mark (verified against MPSKit serial envs)
    Dcod = codomain(y1, 1)
    Ddom = domain(y1, 1)
    data = Array{TT, 3}(undef, 1, nb, 1)
    for b in 1:nb
        y = blocks[b]
        if y === nothing
            # no channel reaches this middle block: exact zero block
            y = zerovector!(similar(y1, (Dcod ⊗ summands[b]) ← Ddom))
        else
            # loud failure (not silent wrong numerics) if the dual/leg
            # conventions of the assembly are off
            space(y, 2) == summands[b] || error(
                "finite-engine env assembly: block $b middle leg $(space(y, 2)) " *
                "does not match expected summand $(summands[b])"
            )
            (codomain(y, 1) == Dcod && domain(y, 1) == Ddom) || error(
                "finite-engine env assembly: block $b virtual legs $(space(y)) " *
                "inconsistent with block $i1 $(space(y1))"
            )
        end
        data[1, b, 1] = y
    end
    sp = ProductSpace(SumSpace(Dcod), Vmid) ←
        ProductSpace(SumSpace(Ddom))
    return BlockTensorMap{TT}(data, sp)
end

function _transfer_left_threaded(GL, W, A)
    pairs = collect(nonzero_pairs(W))
    blocks = _threaded_channel_accumulate(pairs, size(W, 4)) do (I, Wij)
        y = transfer_left(GL.data[1, I[1], 1], Wij, A, A)
        return I[4], y
    end
    # middle leg of a left env built from W = H[j]: W's right virtual leg
    # (stored in the JordanMPOTensor already with the env-ready duality —
    #  verified by the consistency check in _assemble_env)
    return _assemble_env(blocks, space(W, 4))
end

function _transfer_right_threaded(GR, W, A)
    pairs = collect(nonzero_pairs(W))
    blocks = _threaded_channel_accumulate(pairs, size(W, 1)) do (I, Wij)
        y = transfer_right(GR.data[1, I[4], 1], Wij, A, A)
        return I[1], y
    end
    # middle leg of a right env built from W = H[j]: W's left virtual leg
    return _assemble_env(blocks, space(W, 1))
end
