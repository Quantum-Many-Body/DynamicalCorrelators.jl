DefaultDMRG = DMRG2(; tol=1e-8, maxiter=5, verbosity=0,
            alg_eigsolve= Lanczos(;
                krylovdim = 8,
                maxiter = 1,
                tol = 1e-8,
                orth = ModifiedGramSchmidt(),
                eager = true,
                verbosity = 0), 
            alg_svd= SDD(), 
            trscheme=truncdim(4096))


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
            alg_svd = SDD(),
            trscheme=trscheme)