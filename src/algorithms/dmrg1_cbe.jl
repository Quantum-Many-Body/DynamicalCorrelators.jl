"""
    dmrg1_cbe!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector; kwargs...)

1-site DMRG with Controlled Bond Expansion (CBE). At each site, the bond dimension
is first expanded by projecting H|ψ⟩ onto the orthogonal complement (nullspace) of
the current MPS manifold, then a 1-site eigsolve is performed in the enlarged space,
followed by truncation back to the target bond dimension.

# Arguments
- `ψ`: the MPS state (modified in-place)
- `H`: the Hamiltonian (MPOHamiltonian or FiniteMPOHamiltonian)
- `truncdims`: vector of target bond dimensions, one per sweep iteration

# Keyword arguments
- `alg_eigsolve`: eigensolver algorithm (default: `DefaultDMRG1CBE_eigsolve`)
- `alg_svd`: SVD algorithm for CBE expansion (default: `LAPACK_DivideAndConquer()`)
- `δ`: overexpansion fraction (default: `0.1`). At each bond, expand by at least
  `ceil(D * δ)` extra directions beyond target `D`, so that truncation back to `D`
  yields a meaningful discarded weight ξ as error measure.
- `filename`: JLD2 checkpoint file (default: `"default_dmrg1_cbe.jld2"`)
- `verbose`: logging level (default: `true`)
- `envs`: environment cache (default: `environments(ψ, H)`)

# References
- Gleis, Parcollet, Wietek, PRL 130, 246402 (2023)
"""
function dmrg1_cbe!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector;
        alg_eigsolve=DefaultDMRG1CBE_eigsolve,
        alg_svd=LAPACK_DivideAndConquer(),
        δ::Real=0.1,
        filename::String="default_dmrg1_cbe.jld2",
        verbose::Union{Bool, Integer}=true,
        envs=environments(ψ, H))

    N = length(ψ)
    E_prev = real(expectation_value(ψ, H, envs))
    ϵ = 1.0
    start_time, record_start = now(), now()
    timer = TimerOutput()
    Int(verbose) > 0 && println("CBE-DMRG1 Sweep Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
    Int(verbose) > 0 && flush(stdout)

    for iter in eachindex(truncdims)
        alg_eigsolve_iter = updatetol(alg_eigsolve, iter, ϵ)
        trscheme = truncrank(truncdims[iter])
        cbe_ϵp = zeros(Float64, N)
        err² = zeros(Float64, N)

        # ── left to right sweep ──
        for pos in 1:(N - 1)
            @timeit timer "CBE expand" begin
                ϵp, _ = _cbe_expand_l2r!(ψ, H, pos, envs, alg_svd, trscheme, δ; timer=timer)
            end
            cbe_ϵp[pos] = max(cbe_ϵp[pos], ϵp)

            @timeit timer "eigsolve" begin
                h = AC_hamiltonian(pos, ψ, H, ψ, envs)
                _, vecs, _ = eigsolve(h, ψ.AC[pos], 1, :SR, alg_eigsolve_iter)
            end
            ac_new = vecs[1]

            @timeit timer "SVD trunc" begin
                al, S, Vᴴ, ϵ_tr = svd_trunc!(ac_new; trunc=trscheme, alg=alg_svd)
                c = S * Vᴴ
                normalize!(c)
            end
            err²[pos] = ϵ_tr^2

            ψ.AC[pos] = (al, complex(c))
            ψ.AC[pos + 1] = (complex(c), ψ.AR[pos + 1])

            Int(verbose) > 1 && println("  SweepL2R: site $(pos) => site $(pos+1) ", Dates.format(now(), "d.u yyyy HH:MM"))
            Int(verbose) > 1 && flush(stdout)
        end

        # ── right to left sweep ──
        for pos in (N - 1):-1:1
            @timeit timer "CBE expand" begin
                ϵp, _ = _cbe_expand_r2l!(ψ, H, pos, envs, alg_svd, trscheme, δ; timer=timer)
            end
            cbe_ϵp[pos] = max(cbe_ϵp[pos], ϵp)

            @timeit timer "eigsolve" begin
                h = AC_hamiltonian(pos + 1, ψ, H, ψ, envs)
                _, vecs, _ = eigsolve(h, ψ.AC[pos + 1], 1, :SR, alg_eigsolve_iter)
            end
            ac_new = vecs[1]

            @timeit timer "SVD trunc" begin
                U, S, ar_t, ϵ_tr = svd_trunc!(_transpose_tail(ac_new); trunc=trscheme, alg=alg_svd)
                c = U * S
                ar = _transpose_front(ar_t)
                normalize!(c)
            end
            err²[pos] = max(err²[pos], ϵ_tr^2)

            ψ.AC[pos + 1] = (complex(c), ar)
            ψ.AC[pos] = (ψ.AL[pos], complex(c))

            Int(verbose) > 1 && println("  SweepR2L: site $(pos) <= site $(pos+1) ", Dates.format(now(), "d.u yyyy HH:MM"))
            Int(verbose) > 1 && flush(stdout)
        end

        D = N <= 4 ? dim(domain(ψ[N÷2])) : maximum([dim(domain(ψ[N÷2])), dim(domain(ψ[N÷2+1])), dim(domain(ψ[N÷2-1]))])
        E₀ = real(expectation_value(ψ, H, envs))
        ΔE = abs(E₀ - E_prev)
        ϵ = ΔE
        E_prev = E₀
        current_time = now()
        Int(verbose) > 0 && println("[$(iter)/$(length(truncdims))] CBE-DMRG1 sweep", " | duration:", Dates.canonicalize(current_time-start_time))
        Int(verbose) > 0 && println("  E₀ = $(E₀), D = $(D), ΔE = $(ΔE), ϵp = $(maximum(cbe_ϵp)), err² = $(maximum(err²))")
        flush(stdout)
        mode = (iter == 1 ? "w" : "a")
        jldopen(filename, mode) do f
            f["sweep_$(iter)_ψ"] = ψ
            f["sweep_$(iter)_E"]  = E₀
            f["sweep_$(iter)_ΔE"] = ΔE
            f["sweep_$(iter)_cbe_ϵp"] = cbe_ϵp
            f["sweep_$(iter)_err²"] = err²
            f["sweep_$(iter)_D"]  = D
        end
        start_time = current_time
    end
    record_end = now()
    Int(verbose) > 0 && println("Ended: ", Dates.format(record_end, "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(record_end-record_start))
    Int(verbose) > 0 && println(timer)
    return ψ, envs, E_prev
end

"""
    dmrg1_cbe(ψ, H, truncdims; kwargs...)

Non-mutating version of [`dmrg1_cbe!`](@ref). Creates a copy of the MPS `ψ` before
running the CBE-DMRG1 optimization.
"""
function dmrg1_cbe(ψ, H, truncdims; kwargs...)
    return dmrg1_cbe!(copy(ψ), H, truncdims; kwargs...)
end

# ── CBE expansion helpers ──

# Compute how many directions to add: max(target_D - current_D, ceil(current_D * δ))
# Following Gleis et al. PRL 130, 246402: expand to D + D̃ where D̃ = ceil(D * δ),
# then the caller truncates back to target_D.
function _cbe_expand_dim(trscheme, current_D::Int, δ::Real)
    target_D = trscheme.howmany
    D_grow = target_D - current_D             # directions needed to reach target
    D_extra = max(ceil(Int, current_D * δ), 1) # δ-fraction overexpansion
    D_add = max(D_grow, D_extra)
    D_add <= 0 && return nothing
    return truncrank(D_add)
end

"""
    _cbe_expand_l2r!(ψ, H, pos, envs, alg_svd, trscheme, δ) → (ϵp, ϵ_svd) or (0, 0)

CBE expansion at bond (pos, pos+1) during L2R sweep.
Expands by max(target_D - current_D, ceil(current_D * δ)) directions.
"""
function _cbe_expand_l2r!(ψ::AbstractFiniteMPS, H, pos::Int, envs, alg_svd, trscheme, δ::Real; timer=nothing)
    # Determine how many directions to add
    current_D = dim(domain(ψ.AC[pos]))
    cbe_trscheme = _cbe_expand_dim(trscheme, current_D, δ)
    isnothing(cbe_trscheme) && return (0.0, 0.0)

    # Form 2-site tensor and apply effective Hamiltonian
    _ta(f) = isnothing(timer) ? f() : @timeit timer "CBE: build H_ac2+ac2" f()
    H_ac2, ac2 = _ta() do
        (AC2_hamiltonian(pos, ψ, H, ψ, envs), _transpose_front(ψ.AC[pos]) * _transpose_tail(ψ.AR[pos + 1]))
    end
    _th(f) = isnothing(timer) ? f() : @timeit timer "CBE: Hac2*ac2" f()
    Hac2 = _th() do
        H_ac2*ac2
    end

    # Compute left and right nullspaces
    _t2(f) = isnothing(timer) ? f() : @timeit timer "CBE: nullspace" f()
    NL, NR = _t2() do
        NL = left_null(ψ.AC[pos])
        NR = right_null!(_transpose_tail(ψ.AR[pos + 1]; copy=true))
        (NL, NR)
    end

    # Project H|ψ⟩ onto the orthogonal complement and SVD
    _t3(f) = isnothing(timer) ? f() : @timeit timer "CBE: project+SVD" f()
    ϵp, ϵ_svd, V = _t3() do
        intermediate = adjoint(NL) * Hac2 * adjoint(NR)
        ϵp = norm(intermediate)
        ϵp < eps(real(scalartype(ψ)))^(3/4) && return (ϵp, 0.0, nothing)
        normalize!(intermediate)
        _, _, V, ϵ_svd = svd_trunc!(intermediate; trunc=cbe_trscheme, alg=alg_svd)
        (ϵp, ϵ_svd, V)
    end
    isnothing(V) && return (ϵp, 0.0)

    # Construct expansion blocks and update MPS
    _t4(f) = isnothing(timer) ? f() : @timeit timer "CBE: concat+update" f()
    _t4() do
        ar_re = V * NR
        ar_le = zerovector!(similar(ar_re, codomain(ψ.AC[pos]) ← space(V, 1)))
        nal, nc = qr_compact!(catdomain(ψ.AC[pos], ar_le))
        nar = _transpose_front(catcodomain(_transpose_tail(ψ.AR[pos + 1]), ar_re))
        ψ.AC[pos] = (nal, nc)
        ψ.AC[pos + 1] = (nc, nar)
    end

    return (ϵp, ϵ_svd)
end

"""
    _cbe_expand_r2l!(ψ, H, pos, envs, alg_svd, trscheme, δ) → (ϵp, ϵ_svd) or (0, 0)

CBE expansion at bond (pos, pos+1) during R2L sweep.
Expands by max(target_D - current_D, ceil(current_D * δ)) directions.
"""
function _cbe_expand_r2l!(ψ::AbstractFiniteMPS, H, pos::Int, envs, alg_svd, trscheme, δ::Real; timer=nothing)
    # Determine how many directions to add
    current_D = dim(domain(ψ.AL[pos]))
    cbe_trscheme = _cbe_expand_dim(trscheme, current_D, δ)
    isnothing(cbe_trscheme) && return (0.0, 0.0)

    # Form 2-site tensor and apply effective Hamiltonian
    _ta(f) = isnothing(timer) ? f() : @timeit timer "CBE: build H_ac2+ac2" f()
    H_ac2, ac2 = _ta() do
        (AC2_hamiltonian(pos, ψ, H, ψ, envs), _transpose_front(ψ.AL[pos]) * _transpose_tail(ψ.AC[pos + 1]))
    end
    _th(f) = isnothing(timer) ? f() : @timeit timer "CBE: Hac2*ac2" f()
    Hac2 = _th() do
        H_ac2*ac2
    end

    # Compute left and right nullspaces
    _t2(f) = isnothing(timer) ? f() : @timeit timer "CBE: nullspace" f()
    NL, NR = _t2() do
        NL = left_null(ψ.AL[pos])
        NR = right_null!(_transpose_tail(ψ.AC[pos + 1]; copy=true))
        (NL, NR)
    end

    # Project H|ψ⟩ onto the orthogonal complement and SVD
    _t3(f) = isnothing(timer) ? f() : @timeit timer "CBE: project+SVD" f()
    ϵp, ϵ_svd, U_exp = _t3() do
        intermediate = adjoint(NL) * Hac2 * adjoint(NR)
        ϵp = norm(intermediate)
        ϵp < eps(real(scalartype(ψ)))^(3/4) && return (ϵp, 0.0, nothing)
        normalize!(intermediate)
        U, _, _, ϵ_svd = svd_trunc!(intermediate; trunc=cbe_trscheme, alg=alg_svd)
        (ϵp, ϵ_svd, U)
    end
    isnothing(U_exp) && return (ϵp, 0.0)

    # Construct expansion blocks and update MPS
    _t4(f) = isnothing(timer) ? f() : @timeit timer "CBE: concat+update" f()
    _t4() do
        al_re = NL * U_exp
        ac_tail = _transpose_tail(ψ.AC[pos + 1]; copy=true)
        ac_zero = zerovector!(similar(ac_tail, _lastspace(al_re)' ← domain(ac_tail)))
        nal, nc = qr_compact!(catdomain(ψ.AL[pos], al_re))
        nar = _transpose_front(catcodomain(ac_tail, ac_zero))
        ψ.AC[pos] = (nal, nc)
        ψ.AC[pos + 1] = (nc, nar)
    end

    return (ϵp, ϵ_svd)
end