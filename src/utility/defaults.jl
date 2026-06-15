"""
    myDMRG(; tol=1e-8, maxiter=100, krylovdim=16)

Construct the default one-site DMRG algorithm used by this package.

The eigensolver is a one-step Lanczos solver with modified Gram-Schmidt
orthogonalization. Use `tol`, `maxiter`, and `krylovdim` to tune the outer DMRG
stopping tolerance, sweep count, and local Krylov dimension.
"""
myDMRG(;tol=1e-8, maxiter=100, krylovdim=16) = DMRG(; tol=tol, maxiter=maxiter, verbosity=3,
            alg_eigsolve= Lanczos(;
            krylovdim = krylovdim,
            maxiter = 1,
            tol = 1e-8,
            orth = ModifiedGramSchmidt(),
            eager = true,
            verbosity = 0))

"""
    myDMRG2(; tol=1e-6, maxiter=50, trunc=truncrank(4096), krylovdim=16)

Construct the default two-site DMRG algorithm.

`trunc` is passed as the SVD truncation scheme, so callers can use either a
fixed-rank rule such as `truncrank(D)` or a tolerance-based rule. The default
keeps at most 4096 states.
"""
myDMRG2(;tol=1e-6, maxiter=50, trunc=truncrank(4096), krylovdim=16) = DMRG2(; tol=tol, maxiter=maxiter, verbosity=3,
            alg_eigsolve= Lanczos(;
                krylovdim = krylovdim,
                maxiter = 1,
                tol = 1e-8,
                orth = ModifiedGramSchmidt(),
                eager = true,
                verbosity = 0),
            alg_svd= LAPACK_DivideAndConquer(),
            trscheme=trunc)

"""
    myTDVP

Default single-site TDVP algorithm object for finite-MPS time evolution.

This is the fixed-bond-dimension TDVP path from MPSKit, using a Lanczos
integrator with `krylovdim=32`.
"""
myTDVP = TDVP(;
            integrator = Lanczos(;
                krylovdim = 32,
                maxiter = 1,
                tol = 1e-8,
                orth = ModifiedGramSchmidt(),
                eager = true,
                verbosity = 0),
            tolgauge =  1e-13,
            gaugemaxiter = 200)

"""
    myTDVP1_CBE(; D=4096, cbe_tol=1e-10, delta=0.1,
                project_error=false, krylovdim=32)

Construct the default CBE + single-site TDVP algorithm.

`D` is the target bond dimension after each CBE-assisted one-site update.
`delta` controls the temporary CBE overexpansion factor, and `cbe_tol` is the
absolute tolerance used in the CBE selection SVD. Set `project_error=true` to
measure the direct CBE projection error during expansion.
"""
myTDVP1_CBE(; D=4096, cbe_tol=1e-10, delta=0.1, project_error=false, krylovdim=32) = TDVP1_CBE(;
            integrator = Lanczos(;
                krylovdim = krylovdim,
                maxiter = 1,
                tol = 1e-8,
                orth = ModifiedGramSchmidt(),
                eager = true,
                verbosity = 0),
            alg_svd = LAPACK_DivideAndConquer(),
            D = D,
            cbe_tol = cbe_tol,
            delta = delta,
            project_error = project_error)

"""
    myTDVP2(trscheme)

Construct a two-site TDVP algorithm with the given truncation scheme `trscheme`.

Two-site TDVP allows bond-dimension growth through its SVD truncation step and
is commonly used for the first few time steps before switching to `myTDVP` or
`myTDVP1_CBE`.
"""
myTDVP2(trscheme) = TDVP2(;
            integrator = Lanczos(;
                krylovdim = 32,
                maxiter = 1,
                tol = 1e-8,
                orth = ModifiedGramSchmidt(),
                eager = true,
                verbosity = 0),
            tolgauge =  1e-13,
            gaugemaxiter = 200,
            alg_svd = LAPACK_DivideAndConquer(),
            trscheme=trscheme)

"""
    myDMRG1CBE_eigsolve

Default eigensolver for CBE + 1-site DMRG. Uses larger krylovdim than 2-site DMRG
because the 1-site effective Hamiltonian has less variational freedom.
"""
myDMRG1CBE_eigsolve = Lanczos(;
            krylovdim = 16,
            maxiter = 1,
            tol = 1e-8,
            orth = ModifiedGramSchmidt(),
            eager = true,
            verbosity = 0)
