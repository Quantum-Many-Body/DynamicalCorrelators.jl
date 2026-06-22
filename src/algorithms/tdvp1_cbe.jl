"""
    TDVP1_CBE(; integrator=..., alg_svd=..., D=4096,
                cbe_tol=1e-10, delta=0.1, project_error=false,
                finalize=_finalize)

Single-site finite TDVP with direct Controlled Bond Expansion (CBE).

The sweep structure follows MPSKit's finite `TDVP` timestep, while each moving
bond is first enlarged by the direct CBE projector implemented in
`dmrg1_cbe.jl`. After the one-site TDVP update the center is shifted with an
SVD and truncated back to the target bond dimension `D`.

# Keywords
- `integrator`: Krylov time integrator used for `AC` and `C` effective
  evolutions.
- `alg_svd`: SVD backend used for center shifts and CBE selection.
- `D`: target bond dimension after each one-site update.
- `cbe_tol`: absolute tolerance for CBE selection SVDs.
- `delta`: temporary CBE overexpansion factor.
- `project_error`: compute the direct CBE projection error when `true`.
"""
struct TDVP1_CBE{A, S, F} <: Algorithm
    "algorithm used in the exponential solvers"
    integrator::A

    "algorithm used for the singular value decomposition"
    alg_svd::S

    "target bond dimension after each one-site update"
    D::Int

    "absolute tolerance used in the CBE selection SVD"
    cbe_tol::Float64

    "temporary CBE overexpansion factor, so the working target is `D * (1 + delta)`"
    delta::Float64

    "whether to explicitly compute the CBE projection error"
    project_error::Bool

    "callback function applied by `time_evolve` after each timestep"
    finalize::F
end

"""
    TDVP1_CBE(; kwargs...)

Construct a [`TDVP1_CBE`](@ref) algorithm object.

This constructor validates `D`, `cbe_tol`, and `delta`, normalizes the CBE
overexpansion parameter in the same way as `dmrg1_cbe!`, and stores concrete
integrator/SVD choices for use by `timestep` and `timestep!`.
"""
function TDVP1_CBE(;
        integrator = Lanczos(;
            krylovdim = 32,
            maxiter = 1,
            tol = 1e-8,
            orth = ModifiedGramSchmidt(),
            eager = true,
            verbosity = 0
        ),
        alg_svd = LAPACK_DivideAndConquer(),
        D::Integer = 4096,
        cbe_tol::Real = 1e-10,
        delta::Real = 0.1,
        project_error::Bool = false,
        finalize = _finalize
    )
    D > 0 || throw(ArgumentError("D must be a positive integer"))
    cbe_tol >= 0 || throw(ArgumentError("cbe_tol must be nonnegative"))
    return TDVP1_CBE(
        integrator,
        alg_svd,
        Int(D),
        Float64(cbe_tol),
        Float64(_cbe_normalize_delta(delta)),
        project_error,
        finalize
    )
end

function _tdvp1_cbe_shift_right!(ψ::AbstractFiniteMPS, pos::Int, ac, alg::TDVP1_CBE, Dtrunc::Int)
    al, S, Vᴴ, _ = svd_trunc!(ac; trunc=truncrank(Dtrunc), alg=alg.alg_svd)
    c = S * Vᴴ
    ψ.AC[pos] = (al, c)
    ψ.AC[pos + 1] = (c, ψ.AR[pos + 1])
    return c
end

function _tdvp1_cbe_shift_left!(ψ::AbstractFiniteMPS, pos::Int, ac, alg::TDVP1_CBE, Dtrunc::Int)
    U, S, ar_t, _ = svd_trunc!(_transpose_tail(ac); trunc=truncrank(Dtrunc), alg=alg.alg_svd)
    c = U * S
    ar = _transpose_front(ar_t)
    ψ.AC[pos + 1] = (c, ar)
    ψ.AC[pos] = (ψ.AL[pos], c)
    return c
end

"""
    timestep!(ψ, H, t, dt, alg::TDVP1_CBE, envs=environments(ψ, H);
              imaginary_evolution=false, timer=nothing)

Mutating CBE-TDVP1 time step for finite MPS.

The method performs a left-to-right and right-to-left one-site TDVP sweep. Each
moving bond is expanded with the direct CBE projector before the local one-site
evolution, then truncated back to `alg.D` during the center shift. The supplied
`envs` object is reused and returned together with the updated state.
"""
function timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP1_CBE,
        envs = environments(ψ, H);
        imaginary_evolution::Bool = false,
        timer = nothing
    )
    N = length(ψ)
    timer = isnothing(timer) ? TimerOutput() : timer

    @timeit timer "TDVP1_CBE timestep" begin
        @timeit timer "L2R sweep" begin
            for pos in 1:(N - 1)
                Dtrunc = _cbe_effective_target(ψ.AC[pos], ψ.AR[pos + 1], alg.D)
                @timeit timer "CBE expand" begin
                    _cbe_expand_direct_l2r!(
                        ψ, H, pos, envs, alg.alg_svd, alg.D, alg.cbe_tol,
                        alg.delta, alg.project_error, timer
                    )
                end

                Hac = @timeit timer "AC Hamiltonian" AC_hamiltonian(pos, ψ, H, ψ, envs)
                ac = @timeit timer "AC evolution" mps_integrate(
                    Hac, ψ.AC[pos], t, dt / 2, alg.integrator;
                    imaginary_evolution
                )
                c = _tdvp1_cbe_shift_right!(ψ, pos, ac, alg, Dtrunc)

                Hc = @timeit timer "C Hamiltonian" C_hamiltonian(pos, ψ, H, ψ, envs)
                ψ.C[pos] = @timeit timer "C evolution" mps_integrate(
                    Hc, c, t + dt / 2, -dt / 2, alg.integrator;
                    imaginary_evolution
                )
            end
        end

        Hac = AC_hamiltonian(N, ψ, H, ψ, envs)
        ψ.AC[N] = mps_integrate(
            Hac, ψ.AC[N], t, dt / 2, alg.integrator;
            imaginary_evolution
        )

        @timeit timer "R2L sweep" begin
            for site in N:-1:2
                pos = site - 1
                Dtrunc = _cbe_effective_target(ψ.AL[pos], ψ.AC[site], alg.D)
                @timeit timer "CBE expand" begin
                    _cbe_expand_direct_r2l!(
                        ψ, H, pos, envs, alg.alg_svd, alg.D, alg.cbe_tol,
                        alg.delta, alg.project_error, timer
                    )
                end

                Hac = @timeit timer "AC Hamiltonian" AC_hamiltonian(site, ψ, H, ψ, envs)
                ac = @timeit timer "AC evolution" mps_integrate(
                    Hac, ψ.AC[site], t + dt / 2, dt / 2, alg.integrator;
                    imaginary_evolution
                )
                c = _tdvp1_cbe_shift_left!(ψ, pos, ac, alg, Dtrunc)

                Hc = @timeit timer "C Hamiltonian" C_hamiltonian(pos, ψ, H, ψ, envs)
                ψ.C[pos] = @timeit timer "C evolution" mps_integrate(
                    Hc, c, t + dt, -dt / 2, alg.integrator;
                    imaginary_evolution
                )
            end
        end

        Hac = AC_hamiltonian(1, ψ, H, ψ, envs)
        ψ.AC[1] = mps_integrate(
            Hac, ψ.AC[1], t + dt / 2, dt / 2, alg.integrator;
            imaginary_evolution
        )
    end

    return ψ, envs
end

"""
    timestep(ψ, H, t, dt, alg::TDVP1_CBE, envs...; imaginary_evolution=false)

Non-mutating CBE-TDVP1 time step.

Returns a copied or complex-promoted state together with environments, matching
MPSKit's `timestep` convention. Real states are promoted to complex for real
time evolution.
"""
function timestep(
        ψ::AbstractFiniteMPS, H, time::Number, dt::Number,
        alg::TDVP1_CBE, envs...;
        imaginary_evolution::Bool = false, kwargs...
    )
    isreal = (scalartype(ψ) <: Real && !imaginary_evolution)
    ψ′ = isreal ? complex(ψ) : copy(ψ)
    if length(envs) != 0 && isreal
        @warn "Currently cannot reuse real environments for complex evolution"
        envs′ = environments(ψ′, H)
    elseif length(envs) == 1
        envs′ = only(envs)
    else
        @assert length(envs) == 0 "Invalid signature"
        envs′ = environments(ψ′, H)
    end
    return timestep!(ψ′, H, time, dt, alg, envs′; imaginary_evolution, kwargs...)
end
