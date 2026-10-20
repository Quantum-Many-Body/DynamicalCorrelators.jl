# Threaded Krylov-vector kernels for plain `TensorMap`s.
#
# KrylovKit's Lanczos spends the time between matvecs on orthogonalization and
# normalization: `inner`, `add!!`, `scale!!`, `norm` on the full site tensors
# (ModifiedGramSchmidt2 does k inner+axpy pairs per iteration, twice). The
# channel-split Hamiltonian (hamiltonian_threaded.jl) parallelizes the matvec
# across Julia threads, but TensorKit's own implementations of these vector ops
# are serial flat-data loops (UniqueFusion) or serial block loops — so between
# two parallel matvec bursts, a GB-scale orthogonalization runs on ONE core
# while the rest of the node idles. This file closes that gap.
#
# Only the concrete `TensorMap` type is touched; other tensor types
# (BlockTensorMap, DiagonalTensorMap, AdjointTensorMap, ...) keep TensorKit's
# implementations. Two of the methods (`inner`, `norm`) overwrite TensorKit's
# same-signature methods (deliberate type piracy, same pattern as the rest of
# this package's engine overrides); `add!!`/`scale!!` are just more specific
# than TensorKit's AbstractTensorMap fallbacks.
#
# Numerics: identical up to floating-point reduction order (per-thread partial
# sums instead of one serial accumulation), i.e. ~ulp-level differences, and
# results are no longer bit-reproducible across different thread counts.
#
# Scope guard: tensors with fewer than `_VECTOROPS_MIN_LENGTH` scalar entries
# stay on TensorKit's serial path (spawn overhead would exceed the work), and
# non-UniqueFusion symmetries (e.g. SU(2), where `inner` needs dim(c) weights)
# always take TensorKit's serial block loop — threading those is an extension
# point (weighted block loop with dynamic scheduling), currently not needed
# since the production targets are abelian (fℤ₂×U(1)×U(1)).
#
# These kernels carry NO timing instrumentation: they are called from spawned
# tasks as well (KrylovKit's `basistransform!`), where TimerOutputs is not
# thread-safe. The matvec share of the eigensolver is timed at the operator
# level instead (finiteengine/localupdate.jl), and the Gram–Schmidt +
# normalization share is derived by subtraction (algorithms/dmrg.jl).

const _VECTOROPS_THREADED = Ref{Bool}(true)

# scalar-entry count below which the serial path wins
const _VECTOROPS_MIN_LENGTH = 1 << 16

_vectorops_parallel(n::Integer) =
    _VECTOROPS_THREADED[] && Threads.nthreads() > 1 && n >= _VECTOROPS_MIN_LENGTH

# contiguous equal chunks: elementwise kernels have a uniform cost per entry,
# so static partitioning beats dynamic scheduling here (no atomic churn, and
# each thread streams a contiguous memory range)
function _chunk_ranges(n::Int, nt::Int)
    ranges = Vector{UnitRange{Int}}(undef, nt)
    base, rem = divrem(n, nt)
    lo = 1
    for tid in 1:nt
        hi = lo + base + (tid <= rem) - 1
        ranges[tid] = lo:hi
        lo = hi + 1
    end
    return ranges
end

# conj(x)·y over flat data, per-task partial sums merged once at the end
function _flat_inner(x::AbstractVector, y::AbstractVector)
    S = promote_type(eltype(x), eltype(y))
    nt = Threads.nthreads()
    ranges = _chunk_ranges(length(x), nt)
    partials = Vector{S}(undef, nt)
    Threads.@sync for tid in 1:nt
        Threads.@spawn begin
            r = ranges[tid]
            s = zero(S)
            @inbounds @simd for i in r
                s += conj(x[i]) * y[i]
            end
            partials[tid] = s
        end
    end
    return sum(partials)
end

# y .= α .* x .+ β .* y over flat data
function _flat_add!(y::AbstractVector, x::AbstractVector, α::Number, β::Number)
    nt = Threads.nthreads()
    ranges = _chunk_ranges(length(y), nt)
    Threads.@sync for tid in 1:nt
        Threads.@spawn begin
            r = ranges[tid]
            @inbounds @simd for i in r
                y[i] = α * x[i] + β * y[i]
            end
        end
    end
    return y
end

# x .*= α over flat data
function _flat_scale!(x::AbstractVector, α::Number)
    nt = Threads.nthreads()
    ranges = _chunk_ranges(length(x), nt)
    Threads.@sync for tid in 1:nt
        Threads.@spawn begin
            r = ranges[tid]
            @inbounds @simd for i in r
                x[i] *= α
            end
        end
    end
    return x
end

# ---------------------------------------------------------------------------
# VectorInterface / LinearAlgebra overloads (the Lanczos hot path)
# ---------------------------------------------------------------------------

# Julia >= 1.12 forbids overwriting a foreign module's method with the same
# signature during package precompilation, so the two overrides that replace
# TensorKit methods (`inner`, `norm`) are installed at load time from
# `__init__` instead (see DynamicalCorrelators.jl).
const _THREADED_VECTOROPS_INIT = quote

function inner(tx::TensorMap, ty::TensorMap)
    if space(tx) != space(ty)
        throw(SpaceMismatch("$(space(tx)) ≠ $(space(ty))"))
    end
    InnerProductStyle(tx) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:inner)
    if FusionStyle(sectortype(tx)) isa UniqueFusion
        # all quantum dimensions are one: the blocks tile the flat data
        if _vectorops_parallel(length(tx.data))
            return _flat_inner(tx.data, ty.data)
        end
        return inner(tx.data, ty.data)
    end
    # non-abelian: TensorKit's serial block loop with dim(c) weights
    T = promote_inner(tx, ty)
    s = zero(T)
    for c in blocksectors(tx)
        bx = parent(block(tx, c))
        by = parent(block(ty, c))
        s += convert(T, dim(c)) * inner(bx, by)
    end
    return s
end

function norm(t::TensorMap, p::Real = 2)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:norm)
    if FusionStyle(sectortype(t)) isa UniqueFusion
        if p == 2 && _vectorops_parallel(length(t.data))
            return sqrt(real(_flat_inner(t.data, t.data)))
        end
        return norm(t.data, p)
    end
    return _norm(blocks(t), p, float(zero(real(scalartype(t)))))
end

end

function add!!(
        ty::TensorMap, tx::TensorMap, α::Number, β::Number
    )
    T = promote_add(ty, tx, α, β)
    if T <: scalartype(ty)
        if _vectorops_parallel(length(ty.data))
            if space(ty) != space(tx)
                throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
            end
            _flat_add!(ty.data, tx.data, α, β)
            return ty
        end
        return add!(ty, tx, α, β)
    else
        return add(ty, tx, α, β)
    end
end

function scale!!(t::TensorMap, α::Number)
    α === One() && return t
    T = promote_scale(t, α)
    if T <: scalartype(t)
        if _vectorops_parallel(length(t.data))
            _flat_scale!(t.data, α)
        else
            vi_scale!(t.data, α)
        end
        return t
    else
        return scale(t, α)
    end
end

function scale!!(ty::TensorMap, tx::TensorMap, α::Number)
    T = promote_scale(tx, α)
    if T <: scalartype(ty)
        if space(ty) != space(tx)
            throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
        end
        if _vectorops_parallel(length(ty.data))
            nt = Threads.nthreads()
            ranges = _chunk_ranges(length(ty.data), nt)
            Threads.@sync for tid in 1:nt
                Threads.@spawn begin
                    r = ranges[tid]
                    @inbounds @simd for i in r
                        ty.data[i] = α * tx.data[i]
                    end
                end
            end
            return ty
        end
        return vi_scale!(ty, tx, α)
    else
        return scale(tx, α)
    end
end
