# Fast variational compression — Phase 4 of the finite engine.
#
# `fast_approximate!` mirrors MPSKit's `approximate!` (algorithms/approximate/
# fvomps.jl) step for step — same sweep positions, same projection updates,
# same convergence measure — with the sweep running on the mixed
# `HalfFiniteEnvironments(ψ₀, O, ϕ)` instead of MPSKit's full-cache
# `FiniteEnvironments`:
#
#   1. The environments are the mixed overlap environments of ⟨ϕ|O|ψ₀⟩: the
#      transfer takes the fixed above state ϕ on the bra side and the queried
#      below state ψ₀ on the ket side, exactly as MPSKit's `leftenv`/`rightenv`
#      do when `envs.above` is set. Only ψ₀'s tensors are replaced during the
#      sweep, so the === dependency tracking works unchanged.
#   2. The sweep positions coincide with the one-site (DMRG) and two-site
#      (DMRG2) DMRG sweeps, and each local update replaces the same tensors
#      (`ψ.AC[pos] = AC′`, resp. the (al, C)/(C, ar) split) — so the DMRG
#      `_free_after_move!` rules apply verbatim and peak storage stays
#      ~N+O(1) environment tensors instead of 2(N+1).
#   3. The two-site truncated split is routed through the block-parallel
#      `threaded_svd_trunc` whenever Julia threads are available, like the
#      DMRG2 gauge.
#
# The compression operator O is typically a chargedMPO (a plain width-1/2
# FiniteMPO), so the mixed-environment constructor disables the Jordan-MPO
# channel-split threaded transfer — there is nothing to split.

"""
    fast_approximate!(ψ, (O, ϕ), alg::DMRG, envs::HalfFiniteEnvironments;
        verbose = alg.verbosity, manual_gc = true)
    fast_approximate!(ψ, (O, ϕ), alg::DMRG2, envs::HalfFiniteEnvironments;
        verbose = alg.verbosity, manual_gc = true)

Variational compression of `O * ϕ` onto the MPS `ψ` (updated in place),
mirroring MPSKit's `approximate!` with the same single-site (`DMRG`) or
two-site (`DMRG2`) algorithm. Returns `(ψ, envs, ϵ)` with `ϵ` the final
sweep's largest relative update. `envs` must be the mixed environments
`HalfFiniteEnvironments(ψ, O, ϕ)`.

`manual_gc` (default `true`) runs an incremental `GC.gc(false)` after every
local update and a full collection at the end, as in the other fast drivers.
"""
function fast_approximate!(
        ψ::AbstractFiniteMPS, (O, ϕ)::Tuple, alg::DMRG,
        envs::HalfFiniteEnvironments;
        verbose::Int = alg.verbosity, manual_gc::Bool = true
    )
    N = length(ψ)
    allocator = default_allocator(ψ, SerialScheduler())
    ϵ = Inf
    for iter in 1:(alg.maxiter)
        ϵ = 0.0
        for pos in 1:(N - 1)
            AC′ = AC_projection(pos, ψ, O, ϕ, envs; alg.backend, allocator)
            ϵ = max(ϵ, norm(AC′ - ψ.AC[pos]) / norm(AC′))
            ψ.AC[pos] = AC′
            _free_after_move!(envs, alg, Val(:right), pos)
            manual_gc && GC.gc(false)
        end
        for pos in N:-1:2
            AC′ = AC_projection(pos, ψ, O, ϕ, envs; alg.backend, allocator)
            ϵ = max(ϵ, norm(AC′ - ψ.AC[pos]) / norm(AC′))
            ψ.AC[pos] = AC′
            _free_after_move!(envs, alg, Val(:left), pos)
            manual_gc && GC.gc(false)
        end
        verbose >= 3 && println("approximate iter $iter | ϵ = $ϵ")
        if ϵ < alg.tol
            verbose >= 2 && println("approximate converged at iter $iter | ϵ = $ϵ")
            break
        end
        iter == alg.maxiter && verbose >= 1 &&
            @warn "approximate did not converge in $(alg.maxiter) iterations" ϵ
    end
    manual_gc && GC.gc(true)
    return ψ, envs, ϵ
end

function fast_approximate!(
        ψ::AbstractFiniteMPS, (O, ϕ)::Tuple, alg::DMRG2,
        envs::HalfFiniteEnvironments;
        verbose::Int = alg.verbosity, manual_gc::Bool = true
    )
    N = length(ψ)
    allocator = default_allocator(ψ, SerialScheduler())
    alg_gauge = inner_alg_gauge(alg)
    ϵ = Inf
    for iter in 1:(alg.maxiter)
        ϵ = 0.0
        for pos in 1:(N - 1)
            ϵ = max(ϵ, _approximate_split!(ψ, pos, O, ϕ, envs, alg, alg_gauge, allocator))
            _free_after_move!(envs, alg, Val(:right), pos)
            manual_gc && GC.gc(false)
        end
        for pos in (N - 2):-1:1
            ϵ = max(ϵ, _approximate_split!(ψ, pos, O, ϕ, envs, alg, alg_gauge, allocator))
            _free_after_move!(envs, alg, Val(:left), pos)
            manual_gc && GC.gc(false)
        end
        verbose >= 3 && println("approximate iter $iter | ϵ = $ϵ")
        if ϵ < alg.tol
            verbose >= 2 && println("approximate converged at iter $iter | ϵ = $ϵ")
            break
        end
        iter == alg.maxiter && verbose >= 1 &&
            @warn "approximate did not converge in $(alg.maxiter) iterations" ϵ
    end
    manual_gc && GC.gc(true)
    return ψ, envs, ϵ
end

# one two-site projection update of MPSKit's `approximate!(..., ::DMRG2, ...)`:
# project the target onto bond `pos`, truncate back to the single-site
# tensors, return the relative change as the local convergence measure
function _approximate_split!(ψ, pos, O, ϕ, envs, alg, alg_gauge, allocator)
    AC2′ = AC2_projection(pos, ψ, O, ϕ, envs; alg.backend, allocator)
    if alg_gauge isa TruncatedAlgorithm && Threads.nthreads() > 1
        al, c, ar, = threaded_svd_trunc(AC2′, alg_gauge)
    else
        al, c, ar, = svd_trunc!(AC2′, alg_gauge)
    end
    AC2 = ψ.AC[pos] * _transpose_tail(ψ.AR[pos + 1])
    ϵ = norm(al * c * ar - AC2) / norm(AC2)
    ψ.AC[pos] = (al, complex(c))
    ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
    return ϵ
end

"""
    chargedMPS!(ψ, op, gs, site, alg, envs::HalfFiniteEnvironments; kwargs...)
    chargedMPS!(ψ, op, gs, site, alg; disk = false, kwargs...)

In-place form of `chargedMPS(op, gs, site, alg)`: variationally compress
`chargedMPO(op, site, length(gs)) * gs` onto `ψ` (overwritten) with the fast
finite engine. Returns `(ψ, envs, ϵ)` from `fast_approximate!`.

The two-argument-environment form expects `envs = HalfFiniteEnvironments(ψ, O, gs)`
with `O = chargedMPO(op, site, length(gs))`; the keyword form builds it with
the given `disk` backing (`false`/`true`/directory, see [`HalfFiniteEnvironments`](@ref)).
"""
function chargedMPS!(
        ψ::AbstractFiniteMPS, op::AbstractTensorMap, gs::AbstractFiniteMPS,
        site::Integer, alg, envs::HalfFiniteEnvironments; kwargs...
    )
    return fast_approximate!(ψ, (chargedMPO(op, site, length(gs)), gs), alg, envs; kwargs...)
end

function chargedMPS!(
        ψ::AbstractFiniteMPS, op::AbstractTensorMap, gs::AbstractFiniteMPS,
        site::Integer, alg; disk::Union{Bool, AbstractString} = false, kwargs...
    )
    O = chargedMPO(op, site, length(gs))
    envs = HalfFiniteEnvironments(ψ, O, gs; disk)
    return fast_approximate!(ψ, (O, gs), alg, envs; kwargs...)
end

# Zip-up warm start for the variational polish: a single streaming MPO-MPS
# contraction sweep (MPSKit's `Zipup`; Stoudenmire & White, New J. Phys. 12
# (2010); Paeckel et al., Ann. Phys. 411 (2019)) that needs neither
# environments nor an initial guess. The truncation is inherited from the
# polish algorithm's own gauge, so the warm start lands directly on the target
# bond dimension — far better conditioned than the random initial state
# MPSKit's `chargedMPS` uses, and typically converged within a few sweeps.
_default_zipup_alg(alg::DMRG2) = Zipup(inner_alg_gauge(alg))
function _default_zipup_alg(alg::DMRG)
    alg_gauge = inner_alg_gauge(alg)
    alg_gauge isa TruncatedAlgorithm && return Zipup(alg_gauge)
    throw(ArgumentError(
        "cannot derive a zip-up warm start from the non-truncating gauge " *
        "`$(alg_gauge)`: pass an explicit `alg_zipup` (e.g. " *
        "`Zipup(; trunc = truncrank(D))`) to `chargedMPS`, or supply your own " *
        "initial state through `chargedMPS!`."
    ))
end

"""
    chargedMPS(op::AbstractTensorMap, gs::AbstractFiniteMPS, site::Integer, alg; disk = false, alg_zipup = ...)

Approximate `chargedMPO(op, site, length(gs)) * gs` with the supplied MPSKit
algorithm `alg` (single-site `DMRG` or two-site `DMRG2`), running on the fast
finite engine. The initial state is a zip-up warm start: the MPO-MPS product
is contracted in a single streaming sweep and truncated on the fly with
`alg`'s own truncated gauge, instead of the random state used by MPSKit's
`chargedMPS`. Override the warm start with `alg_zipup` (e.g. a Paeckel-style
two-pass `Zipup(; trunc = (truncrank(2D), truncrank(D)))`, required for a
`DMRG` polish whose gauge does not truncate), or bring a fully custom initial
state through [`chargedMPS!`](@ref). Pass `disk = true` (or a directory) to
back the compression environments by disk instead of RAM.
"""
function chargedMPS(
        op::AbstractTensorMap, gs::AbstractFiniteMPS, site::Integer, alg;
        disk::Union{Bool, AbstractString} = false,
        alg_zipup = _default_zipup_alg(alg),
        kwargs...
    )
    ψ₀, = approximate((chargedMPO(op, site, length(gs)), gs), alg_zipup)
    ψ, = chargedMPS!(ψ₀, op, gs, site, alg; disk, kwargs...)
    return ψ
end
