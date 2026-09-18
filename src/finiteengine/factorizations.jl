# Block-parallel truncated SVD and gauge steps — Phase 3+ of the finite engine.
#
# TensorKit routes factorizations through MatrixAlgebraKit (MAK); the tensor
# level `foreachblock` loop is serial (its scheduler hook is an upstream TODO).
# For SU(2) states at large D the gauge-step SVD has many independent sector
# blocks, so here the per-sector `svd_compact!` calls are distributed over
# Julia threads.
#
# CRITICAL: only the per-block LAPACK calls are parallelized. The truncation
# itself stays EXACTLY TensorKit's: for SU(2), `truncrank` sorts *globally*
# across blocks weighted by quantum dimensions (TensorKit truncation.jl,
# `findtruncated(::SectorVector, ::TruncationByOrder)`), so truncating block by
# block would be wrong. The flow below mirrors MAK's generic `svd_trunc!`:
#   initialize_output(svd_compact!) → per-block svd_compact! (threaded)
#   → MAK.truncate(svd_trunc!, F, strategy) (global) → truncation_error.

"""
    threaded_svd_trunc(t::AbstractTensorMap, alg::TruncatedAlgorithm)

Block-parallel drop-in for `TensorKit.svd_trunc(t, alg)`: per-sector compact
SVDs run on Julia threads, the (global, quantum-dimension-weighted) truncation
and the error `ϵ` are computed by TensorKit/MatrixAlgebraKit exactly as in the
serial path. Falls back to `svd_trunc` on a single thread.
"""
function threaded_svd_trunc(t::AbstractTensorMap, alg::TruncatedAlgorithm)
    (Threads.nthreads() == 1 || length(blocks(t)) <= 1) && return svd_trunc(t, alg)
    # per-block svd_compact! (LAPACK gesdd) DESTROYS the input blocks; work on a
    # copy so the caller's tensor is never mutated (MAK's serial out-of-place
    # path copies for the same reason)
    t = copy(t)
    F = initialize_output(svd_compact!, t, alg.alg)
    _threaded_svd_compact_blocks!(t, F, alg.alg)
    (U, S, Vᴴ), ind = truncate(svd_trunc!, F, alg.trunc)
    ϵ = truncation_error(diagview(F[2]), ind)
    return U, S, Vᴴ, ϵ
end

# Per-sector `svd_compact!` over Julia threads. Each task touches disjoint
# block memory; dynamic scheduling because block sizes vary wildly. Mirrors
# TensorKit's serial `foreachblock` loop in matrixalgebrakit.jl, including the
# out-of-place (`copy!`) guard.
function _threaded_svd_compact_blocks!(t::AbstractTensorMap, F, alg)
    items = collect(blocks(t))
    Threads.@threads :dynamic for i in eachindex(items)
        c, tb = items[i]
        Fb = map(f -> block(f, c), F)
        Fb′ = svd_compact!(tb, Fb, alg)
        for (b′, b) in zip(Fb′, Fb)
            b === b′ || copy!(b, b′)
        end
    end
    return F
end

# ---------------------------------------------------------------------------
# threaded gauge steps — mirrors of MPSKit's orthoview.jl implementations with
# the block-parallel SVD swapped in (only for a plain TruncatedAlgorithm
# gauge; other gauges keep MPSKit's serial path)
# ---------------------------------------------------------------------------

# orthoview.jl `left_gauge(AC, ::TruncatedAlgorithm)`
function _threaded_left_gauge(AC, alg::TruncatedAlgorithm)
    U, S, Vᴴ, ϵ = threaded_svd_trunc(AC, alg)
    C = lmul!(S, Vᴴ) # C = S * Vᴴ, matching `LeftOrthViaSVD`
    return U, C, ϵ
end

# orthoview.jl `right_gauge(AC, ::TruncatedAlgorithm)`
function _threaded_right_gauge(AC, alg::TruncatedAlgorithm)
    U, S, Vᴴ, ϵ = threaded_svd_trunc(_transpose_tail(AC), alg)
    C = rmul!(U, S) # C = U * S, matching `RightOrthViaSVD`
    return C, _transpose_front(Vᴴ), ϵ
end

# orthoview.jl `left_gauge!` / `right_gauge!`
function _threaded_gauge!(ψ::AbstractFiniteMPS, pos::Int, ::Val{:right}, AC, alg::TruncatedAlgorithm; normalize::Bool)
    AL, C, ϵ = _threaded_left_gauge(AC, alg)
    normalize && normalize!(C)
    ψ.AC[pos] = (AL, C)
    return ψ, ϵ
end
function _threaded_gauge!(ψ::AbstractFiniteMPS, pos::Int, ::Val{:left}, AC, alg::TruncatedAlgorithm; normalize::Bool)
    C, AR, ϵ = _threaded_right_gauge(AC, alg)
    normalize && normalize!(C)
    ψ.AC[pos] = (C, AR)
    return ψ, ϵ
end

# orthoview.jl `gauge2!` (two-site split of the updated AC2)
function _threaded_gauge2!(ψ::AbstractFiniteMPS, pos::Int, ::Val{Dir}, AC2, alg::TruncatedAlgorithm; normalize::Bool) where {Dir}
    al, c, ar, ϵ = threaded_svd_trunc(AC2, alg)
    normalize && normalize!(c)
    C = scalartype(ψ) <: Complex ? complex(c) : c
    if Dir === :right
        ψ.AC[pos] = (al, C)
        ψ.AC[pos + 1] = (C, _transpose_front(ar))
    elseif Dir === :left
        ψ.AC[pos + 1] = (C, _transpose_front(ar))
        ψ.AC[pos] = (al, C)
    else
        throw(ArgumentError(lazy"invalid direction `$Dir`"))
    end
    return ψ, ϵ
end
