"""
    DefaultDMRG

Default DMRG2 algorithm configuration for ground-state optimization.
Uses Lanczos eigensolver with `krylovdim=3`, LAPACK SVD, and bond dimension truncation at 4096.
"""
DefaultDMRG = DMRG2(; tol=1e-8, maxiter=5, verbosity=0,
            alg_eigsolve= Lanczos(;
                krylovdim = 3,
                maxiter = 1,
                tol = 1e-8,
                orth = ModifiedGramSchmidt(),
                eager = true,
                verbosity = 0),
            alg_svd= LAPACK_DivideAndConquer(),
            trscheme=truncrank(4096))

"""
    DefaultDMRG2(tol, krylovdim)

Construct a DMRG2 algorithm with custom eigensolver tolerance `tol` and Krylov dimension `krylovdim`.
Other parameters are the same as [`DefaultDMRG`](@ref).
"""
DefaultDMRG2(tol, krylovdim) = DMRG2(; tol=1e-8, maxiter=5, verbosity=0,
            alg_eigsolve= Lanczos(;
                krylovdim = krylovdim,
                maxiter = 1,
                tol = tol,
                orth = ModifiedGramSchmidt(),
                eager = true,
                verbosity = 0),
            alg_svd= LAPACK_DivideAndConquer(),
            trscheme=truncrank(4096))


"""
    DefaultTDVP

Default single-site TDVP algorithm configuration for time evolution.
Uses Lanczos integrator with `krylovdim=32` for accurate time-stepping.
"""
DefaultTDVP = TDVP(;
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
    DefaultTDVP2(trscheme)

Construct a two-site TDVP algorithm with the given truncation scheme `trscheme`.
Two-site TDVP allows bond dimension growth and is used for the first few time steps.
"""
DefaultTDVP2(trscheme) = TDVP2(;
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
