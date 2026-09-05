"""
    myDMRG1(; tol=1e-6, maxiter=100, krylovdim=16, adaptive=true, alg_eigsolve=nothing)

Construct the default one-site DMRG algorithm used by this package.

The local eigensolver is adaptive by default (`AdaptiveKrylov`: tolerance,
Krylov dimension and restart count are retuned per local update from the
measured convergence behavior). Set `adaptive = false` to pin a one-step
`Lanczos` with the given `krylovdim`, or pass `alg_eigsolve` explicitly for
full manual control. `tol` and `maxiter` tune the outer DMRG stopping criterion
and sweep count.
"""
myDMRG1(; tol=1e-6, maxiter=100, krylovdim=16, adaptive::Bool=true, alg_eigsolve=nothing) =
    DMRG(; tol=tol, maxiter=maxiter, verbosity=3,
            alg_eigsolve = _resolve_alg_eigsolve(alg_eigsolve, adaptive, krylovdim))

"""
    myDMRG2(; tol=1e-6, maxiter=50, trunc=truncrank(4096), krylovdim=16, adaptive=true, alg_eigsolve=nothing)

Construct the default two-site DMRG algorithm.

`trunc` is passed as the SVD truncation scheme, so callers can use either a
fixed-rank rule such as `truncrank(D)` or a tolerance-based rule. The default
keeps at most 4096 states. The local eigensolver is adaptive by default
(`AdaptiveKrylov`); set `adaptive = false` to pin a one-step `Lanczos` with the
given `krylovdim`, or pass `alg_eigsolve` explicitly for full manual control.
"""
myDMRG2(; tol=1e-6, maxiter=50, trunc=truncrank(4096), krylovdim=16, adaptive::Bool=true, alg_eigsolve=nothing) =
    DMRG2(; tol=tol, maxiter=maxiter, verbosity=3,
            alg_eigsolve = _resolve_alg_eigsolve(alg_eigsolve, adaptive, krylovdim),
            alg_svd= LAPACK_DivideAndConquer(),
            trunc=trunc)

"""
    myTDVP1(; krylovdim=24)

Construct the default single-site TDVP algorithm for finite-MPS time evolution.

This is the fixed-bond-dimension TDVP path from MPSKit. `krylovdim` controls the
dimension of the Lanczos Krylov subspace used by the local time integrator.
Unlike the DMRG eigensolvers, the time integrator is intentionally *not*
adaptive: MPSKit's `integrate` dispatches on a concrete `Lanczos`/`Arnoldi`,
and the required Krylov dimension (roughly `dt` x spectral width) barely changes
during the evolution. Watch for "integrator failed to converge" warnings when
taking large `dt`.
"""
myTDVP1(; krylovdim = 24) = TDVP(;
            integrator = Lanczos(;
                krylovdim = krylovdim,
                maxiter = 1,
                tol = 1e-8,
                orth = ModifiedGramSchmidt(),
                eager = true,
                verbosity = 0),
            tolgauge =  1e-13,
            gaugemaxiter = 200)

"""
    myTDVP1_CBE(; D=4096, delta=0.1, krylovdim=30)

Construct the default single-site TDVP algorithm with Controlled Bond Expansion.

This is a thin wrapper around MPSKit's `TDVP` with
`alg_expand = OptimalExpand(...)`: ahead of each one-site evolution the moving
bond is enlarged by up to `ceil(Int, delta*D)` directions selected from the
projected two-site update, and the truncating gauge (`trunc = truncrank(D)`)
cuts the enlarged bond back to `D` when the center moves. Note that the `trunc`
of `OptimalExpand` counts the directions *added* per bond, so the former
overexpansion factor `delta` enters through `ceil(Int, delta*D)`.
"""
myTDVP1_CBE(; D=4096, delta=0.1, krylovdim=30) = TDVP(;
            integrator = Lanczos(;
                krylovdim = krylovdim,
                maxiter = 1,
                tol = 1e-8,
                orth = ModifiedGramSchmidt(),
                eager = true,
                verbosity = 0),
            alg_expand = OptimalExpand(;
                alg_svd = LAPACK_DivideAndConquer(),
                trunc = truncrank(ceil(Int, delta*D))),
            alg_svd = LAPACK_DivideAndConquer(),
            trunc = truncrank(D))

"""
    myTDVP2(; trunc=truncrank(4096), krylovdim=30)

Construct the default two-site TDVP algorithm.

`trunc` is passed as the SVD truncation scheme, so callers can use either a
fixed-rank rule such as `truncrank(D)` or a tolerance-based rule. `krylovdim`
controls the Lanczos Krylov dimension for the two-site time integrator.
"""
myTDVP2(; trunc = truncrank(4096), krylovdim = 30) = TDVP2(;
            integrator = Lanczos(;
                krylovdim = krylovdim,
                maxiter = 1,
                tol = 1e-8,
                orth = ModifiedGramSchmidt(),
                eager = true,
                verbosity = 0),
            tolgauge =  1e-13,
            gaugemaxiter = 200,
            alg_svd = LAPACK_DivideAndConquer(),
            trunc=trunc)

"""
    myDMRG1_CBE(; tol=1e-6, maxiter=100, D=4096, delta=0.1, krylovdim=16, adaptive=true, alg_eigsolve=nothing)

Construct the default one-site DMRG algorithm with Controlled Bond Expansion.

This is a thin wrapper around MPSKit's `DMRG` with
`alg_expand = OptimalExpand(...)`: ahead of each one-site eigensolve the moving
bond is enlarged by up to `ceil(Int, delta*D)` directions selected from the
projected two-site update, and the truncating gauge (`trunc = truncrank(D)`)
cuts the enlarged bond back to `D` when the center moves. Note that the `trunc`
of `OptimalExpand` counts the directions *added* per bond, so the former
overexpansion factor `delta` enters through `ceil(Int, delta*D)`. The local
eigensolver is adaptive by default (`AdaptiveKrylov`); set `adaptive = false`
to pin a one-step `Lanczos` with the given `krylovdim`, or pass `alg_eigsolve`
explicitly for full manual control.
"""
myDMRG1_CBE(; tol=1e-6, maxiter=100, D=4096, delta=0.1, krylovdim=16, adaptive::Bool=true, alg_eigsolve=nothing) = DMRG(;
            tol = tol,
            maxiter = maxiter,
            verbosity = 3,
            alg_eigsolve = _resolve_alg_eigsolve(alg_eigsolve, adaptive, krylovdim),
            alg_expand = OptimalExpand(;
                alg_svd = LAPACK_DivideAndConquer(),
                trunc = truncrank(ceil(Int, delta*D))),
            alg_svd = LAPACK_DivideAndConquer(),
            trunc = truncrank(D))
