"""
    dmrg1_cbe!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector; kwargs...)

1-site DMRG with Controlled Bond Expansion (CBE) following Algorithm 1 of
Gleis, Li, von Delft, PRL 130, 246402 (2023).

At each bond, the kept space A_ℓ is expanded by a truncated orthogonal complement
Â_ℓ^tr via shrewd selection (preselection + final selection), both at 1-site cost.
Then 1-site eigsolve is performed in the enlarged space, followed by truncation
back to the target bond dimension.

# Arguments
- `ψ`: the MPS state (modified in-place)
- `H`: the Hamiltonian (MPOHamiltonian)
- `truncdims`: vector of target bond dimensions D_f, one per sweep iteration

# Keyword arguments
- `alg_eigsolve`: eigensolver algorithm (default: `DefaultDMRG1CBE_eigsolve`)
- `alg_svd`: SVD algorithm (default: `LAPACK_DivideAndConquer()`)
- `cbe_tol`: absolute truncation tolerance used in CBE selection SVDs
  together with their rank cutoffs (default: `1e-10`)
- `delta`: CBE working-space overexpansion. The bond is expanded from `D_i`
  to `D_f*(1+delta)` before the eigensolver, then truncated back to `D_f`
  when shifting the center (default: `0.1`)
- `preselect_factor`: preselection factor. Use `:none` for `D′=D_f`, or a real
  value for `D′=preselect_factor*D_f/w*`, where `w*` counts MPO-bond symmetry
  multiplets (default: `1.0`, i.e. moderate preselection).
- `safety`: check step-c orthogonality and reorthogonalize only if needed
  (default: `false`)
- `project_error`: explicitly compute the CBE projection error
  `|A_l A_{l+1} - A_l^ex A_{l+1}^ex|`; if `false`, report `NaN`
  (default: `false`)
- `filename`: JLD2 checkpoint file (default: `"default_dmrg1_cbe.jld2"`)
- `save_iters`: which sweep iterations to save (default: all)
- `verbose`: logging level (default: `true`)
- `envs`: environment cache (default: `environments(ψ, H)`)

# References
- Gleis, Li, von Delft, Phys. Rev. Lett. 130, 246402 (2023)
"""
function dmrg1_cbe!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector;
        alg_eigsolve=DefaultDMRG1CBE_eigsolve,
        alg_svd=LAPACK_DivideAndConquer(),
        cbe_tol::Real=1e-10,
        delta::Real=0.1,
        preselect_factor::Union{Symbol, Real}=1.0,
        safety::Bool=false,
        project_error::Bool=false,
        filename::String="default_dmrg1_cbe.jld2",
        save_iters::AbstractVector{<:Integer}=1:length(truncdims),
        verbose::Union{Bool, Integer}=true,
        envs=environments(ψ, H))

    N = length(ψ)
    delta = _cbe_normalize_delta(delta)
    preselect_factor = _cbe_normalize_preselect_factor(preselect_factor)
    E_prev = real(expectation_value(ψ, H, envs))
    ϵ = 1.0
    start_time, record_start = now(), now()
    timer = TimerOutput()
    # formatting widths
    wpos  = ndigits(N)
    witer = ndigits(length(truncdims))
    wD    = max(ndigits(_cbe_work_target(maximum(truncdims), delta)), 4)
    Int(verbose) > 0 && println("CBE-DMRG1 Sweep Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
    Int(verbose) > 0 && flush(stdout)

    for iter in eachindex(truncdims)
        alg_eigsolve_iter = updatetol(alg_eigsolve, iter, ϵ)
        cbe_ϵp = fill(NaN, N)
        cbe_nonorth_l2r = Int[]
        cbe_nonorth_r2l = Int[]
        err² = zeros(Float64, N)

        # ── left to right sweep ──
        # Expand the left tensor on bond (pos,pos+1), optimize the enlarged
        # center at pos+1, then push the center back to the left.
        for pos in 1:(N - 1)
            Dtrunc = _cbe_effective_target(ψ.AL[pos], ψ.AR[pos + 1], truncdims[iter])
            @timeit timer "CBE expand" begin
                ϵp, failed_nonorth = _cbe_expand_l2r!(ψ, H, pos, envs, alg_svd, truncdims[iter], cbe_tol, delta, preselect_factor, safety, project_error, timer)
            end
            cbe_ϵp[pos] = _cbe_nanmax(cbe_ϵp[pos], ϵp)
            failed_nonorth && push!(cbe_nonorth_l2r, pos)

            @timeit timer "eigsolve" begin
                h = AC_hamiltonian(pos + 1, ψ, H, ψ, envs)
                _, vecs, _ = eigsolve(h, ψ.AC[pos + 1], 1, :SR, alg_eigsolve_iter)
            end
            ac_new = vecs[1]

            @timeit timer "SVD trunc" begin
                # move center left: AC[pos+1] → (U, S, ar), truncate bond back to target D
                U, S, ar_t, ϵ_tr = svd_trunc!(_transpose_tail(ac_new); trunc=truncrank(Dtrunc), alg=alg_svd)
                ar = _transpose_front(ar_t)
                # absorb U into AL: AL domain D+D̃ → D_f
                @plansor new_AL[-1 -2; -3] := ψ.AL[pos][-1 -2; 1] * U[1; -3]
                normalize!(S)
            end
            err²[pos] = ϵ_tr^2

            ψ.AC[pos]     = (new_AL, S)
            ψ.AC[pos + 1] = (S, ar)

            Int(verbose) > 1 && println("  SweepL2R: site ", lpad(pos, wpos), " => site ", lpad(pos+1, wpos),
                " | D = ", lpad(dim(codomain(ψ.AC[pos+1])[1]), wD),
                " | ", Dates.format(now(), "d.u yyyy HH:MM"))
            Int(verbose) > 1 && flush(stdout)
        end
        @timeit timer "eigsolve" begin
            h = AC_hamiltonian(N, ψ, H, ψ, envs)
            _, vecs, _ = eigsolve(h, ψ.AC[N], 1, :SR, alg_eigsolve_iter)
        end
        ψ.AC[N] = normalize!(vecs[1])

        # ── right to left sweep ──
        # Expand the right tensor on bond (pos,pos+1), optimize the enlarged
        # center at pos, then push the center back to the right.
        for pos in (N - 1):-1:1
            Dtrunc = _cbe_effective_target(ψ.AL[pos], ψ.AR[pos + 1], truncdims[iter])
            @timeit timer "CBE expand" begin
                ϵp, failed_nonorth = _cbe_expand_r2l!(ψ, H, pos, envs, alg_svd, truncdims[iter], cbe_tol, delta, preselect_factor, safety, project_error, timer)
            end
            cbe_ϵp[pos] = _cbe_nanmax(cbe_ϵp[pos], ϵp)
            failed_nonorth && push!(cbe_nonorth_r2l, pos)

            @timeit timer "eigsolve" begin
                h = AC_hamiltonian(pos, ψ, H, ψ, envs)
                _, vecs, _ = eigsolve(h, ψ.AC[pos], 1, :SR, alg_eigsolve_iter)
            end
            ac_new = vecs[1]

            @timeit timer "SVD trunc" begin
                # move center right: AC[pos] → (al, S, Vᴴ), truncate bond back to target D
                al, S, Vᴴ, ϵ_tr = svd_trunc!(ac_new; trunc=truncrank(Dtrunc), alg=alg_svd)
                # absorb Vᴴ into AR: AR codomain D+D̃ → D_f
                @plansor new_AR[-1 -2; -3] := Vᴴ[-1; 1] * ψ.AR[pos + 1][1 -2; -3]
                normalize!(S)
            end
            err²[pos] = max(err²[pos], ϵ_tr^2)

            ψ.AC[pos]     = (al, S)
            ψ.AC[pos + 1] = (S, new_AR)

            Int(verbose) > 1 && println("  SweepR2L: site ", lpad(pos, wpos), " <= site ", lpad(pos+1, wpos),
                " | D = ", lpad(dim(domain(ψ.AC[pos])[1]), wD),
                " | ", Dates.format(now(), "d.u yyyy HH:MM"))
            Int(verbose) > 1 && flush(stdout)
        end
        @timeit timer "eigsolve" begin
            h = AC_hamiltonian(1, ψ, H, ψ, envs)
            _, vecs, _ = eigsolve(h, ψ.AC[1], 1, :SR, alg_eigsolve_iter)
        end
        ψ.AC[1] = normalize!(vecs[1])

        D = _cbe_max_bond_dimension(ψ)
        E₀ = real(expectation_value(ψ, H, envs))
        ΔE = abs(E₀ - E_prev)
        ϵ = ΔE
        E_prev = E₀
        current_time = now()
        Int(verbose) > 0 && println("[", lpad(iter, witer), "/", length(truncdims), "] CBE-DMRG1 sweep | duration: ",
            Dates.canonicalize(current_time-start_time))
        Int(verbose) > 0 && @printf("  E₀ = %.10f | D = %*d | ΔE = %.3e | ϵp = %.3e | err² = %.3e\n",
            E₀, wD, D, ΔE, _cbe_nanmax(cbe_ϵp), maximum(err²))
        if Int(verbose) > 0 && (!isempty(cbe_nonorth_l2r) || !isempty(cbe_nonorth_r2l))
            println("  CBE nonorth skipped bonds: L2R=", cbe_nonorth_l2r,
                " | R2L=", cbe_nonorth_r2l)
        end
        flush(stdout)
        if iter in save_iters
            mode = (iter == first(save_iters) ? "w" : "a")
            jldopen(filename, mode) do f
                f["sweep_$(iter)_ψ"] = ψ
                f["sweep_$(iter)_E"]  = E₀
                f["sweep_$(iter)_ΔE"] = ΔE
                f["sweep_$(iter)_cbe_ϵp"] = cbe_ϵp
                f["sweep_$(iter)_err²"] = err²
                f["sweep_$(iter)_D"]  = D
            end
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

Non-mutating version of [`dmrg1_cbe!`](@ref).
"""
function dmrg1_cbe(ψ, H, truncdims; kwargs...)
    return dmrg1_cbe!(copy(ψ), H, truncdims; kwargs...)
end

function _cbe_max_bond_dimension(ψ::AbstractFiniteMPS)
    N = length(ψ)
    N <= 1 && return 1
    return maximum(pos -> dim(domain(ψ.AL[pos])[1]), 1:(N - 1))
end

_cbe_work_target(D_f::Integer, delta::Real) = max(Int(D_f), ceil(Int, (1 + delta) * Int(D_f)))

_cbe_preselect_trunc(maxdim::Integer, tol::Real) = truncrank(maxdim) & trunctol(atol=tol)
_cbe_preselect_orth_tol() = 1e-8

_cbe_nanmax(a::Real, b::Real) = isnan(a) ? Float64(b) : (isnan(b) ? Float64(a) : max(Float64(a), Float64(b)))
_cbe_nanmax(xs::AbstractVector{<:Real}) = isempty(xs) ? NaN : foldl(_cbe_nanmax, xs; init=NaN)

function _cbe_normalize_delta(delta::Real)
    delta >= 0 && return delta
    throw(ArgumentError("delta must be a nonnegative real overexpansion factor"))
end

function _cbe_normalize_preselect_factor(preselect_factor::Union{Symbol, Real})
    preselect_factor === :none && return preselect_factor
    preselect_factor isa Real && preselect_factor >= 0 && return preselect_factor
    throw(ArgumentError("preselect_factor must be :none or a nonnegative real preselection factor"))
end

function _cbe_preselect_rank(D_f::Integer, wstar::Integer, preselect_factor::Union{Symbol, Real})
    preselect_factor === :none && return Int(D_f)
    return max(floor(Int, preselect_factor * Int(D_f) / Int(wstar)), 1)
end

function _cbe_effective_target(AL, AR, D_f::Integer)
    Dl = dim(codomain(AL))
    Dr = _cbe_right_capacity(AR)
    return min(Int(D_f), Dl, Dr)
end

function _cbe_right_capacity(AR)
    dphys = length(codomain(AR)) <= 1 ? 1 : mapreduce(i -> dim(codomain(AR)[i]), *, 2:length(codomain(AR)); init=1)
    return dphys * dim(domain(AR))
end

_cbe_multipletdim(V) = sum(c -> dim(V, c), sectors(V); init=0)

# ════════════════════════════════════════════════════════════════════════════
#  CBE expansion
# ════════════════════════════════════════════════════════════════════════════

"""
    _cbe_expand_l2r!(ψ, H, pos, envs, alg_svd, trscheme, delta) → ϵp

Top-level orchestrator for CBE bond expansion at bond (pos, pos+1), L→R sweep.

Calls the four Fig. S-2 sub-steps in order, then sets ψ.AL[pos] = A^ex and
ψ.AC[pos+1] = C^{e.i.}_{ℓ+1} (zero-padded). Returns ϵp, the explicit
projection error, if requested; otherwise returns `NaN`.
"""
function _cbe_expand_l2r!(ψ::AbstractFiniteMPS, H, pos::Int, envs, alg_svd, D_f, cbe_tol::Real, delta::Real, preselect_factor::Union{Symbol, Real}, safety::Bool, project_error::Bool, timer::TimerOutput)
    AL = ψ.AL[pos]
    AR = ψ.AR[pos + 1]
    Λ  = ψ.C[pos]

    D_eff = _cbe_effective_target(AL, AR, _cbe_work_target(D_f, delta))
    D_current = dim(domain(AL)[1])
    D_tilde = D_eff - D_current
    D_tilde <= 0 && return NaN, false

    GL = leftenv(envs, pos, ψ)
    GR = rightenv(envs, pos + 1, ψ)
    W1 = H[pos]; W2 = H[pos + 1]
    w = _cbe_multipletdim(domain(W1)[2])
    D_prime = _cbe_preselect_rank(D_f, w, preselect_factor)

    # (a) TABLE I steps 1-6 (GETRORTH): R^orth_ℓ = (I - B B†)(W_{ℓ+1} R_{ℓ+2}), SVD → U, S
    U, S, _ = @timeit timer "CBE step a" _cbe_getrorth_l2r(GR, W2, AR, Λ)

    # (b) TABLE I steps 7-12 (GETLORTH): L^orth_ℓ = (I - A A†)(GL W_ℓ U S), SVD-truncate D→D'
    L_orth_tr = @timeit timer "CBE step b" _cbe_getlorth_l2r(GL, W1, AL, U, S, D_prime, alg_svd, cbe_tol)

    # (c) TABLE I steps 13-15: redirect MPO leg, SVD → Â^pr (image dim D̂ = D_f)
    A_pr = @timeit timer "CBE step c" _cbe_get_Apr_l2r(L_orth_tr, AL, D_eff, alg_svd, cbe_tol, safety)
    isnothing(A_pr) && return NaN, true

    # (d) TABLE I steps 16-23: GETCORTH + final SVD → Â^tr (image dim D̃)
    A_tr = @timeit timer "CBE step d" _cbe_get_Atr_l2r(A_pr, AL, AR, Λ, W1, W2, GL, GR, D_tilde, alg_svd, cbe_tol)

    # Bond expansion: A^ex_ℓ = A_ℓ ⊕ Â^tr_ℓ, C^{e.i.}_{ℓ+1} zero-padded
    ϵp = _cbe_bond_expand_l2r!(ψ, pos, AL, A_tr, project_error)

    return ϵp, false
end

"""
    _cbe_expand_r2l!(ψ, H, pos, envs, alg_svd, trscheme, delta) → ϵp

Top-level orchestrator for CBE bond expansion at bond (pos, pos+1), R→L sweep.
Mirror of `_cbe_expand_l2r!`: builds B̂^tr_{ℓ+1} and expands from the right.
"""
function _cbe_expand_r2l!(ψ::AbstractFiniteMPS, H, pos::Int, envs, alg_svd, D_f, cbe_tol::Real, delta::Real, preselect_factor::Union{Symbol, Real}, safety::Bool, project_error::Bool, timer::TimerOutput)
    AL = ψ.AL[pos]
    AR = ψ.AR[pos + 1]
    Λ  = ψ.C[pos]

    D_eff = _cbe_effective_target(AL, AR, _cbe_work_target(D_f, delta))
    D_current = dim(codomain(AR)[1])
    D_tilde = D_eff - D_current
    D_tilde <= 0 && return NaN, false

    GL = leftenv(envs, pos, ψ)
    GR = rightenv(envs, pos + 1, ψ)
    W1 = H[pos]; W2 = H[pos + 1]
    w = _cbe_multipletdim(codomain(W2)[1])
    D_prime = _cbe_preselect_rank(D_f, w, preselect_factor)

    # (a) TABLE I steps 1-6 (mirrored, GETLORTH): L^orth_{ℓ} = (I - A A†)(GL W_ℓ Λ), SVD → _, S, Vᴴ
    _, S, Vᴴ = @timeit timer "CBE step a" _cbe_getlorth_r2l(GL, W1, AL, Λ)

    # (b) TABLE I steps 7-12 (mirrored, GETRORTH): R^orth_{ℓ+1}, SVD-truncate D → D'
    R_orth_tr = @timeit timer "CBE step b" _cbe_getrorth_r2l(GR, W2, AR, S, Vᴴ, D_prime, alg_svd, cbe_tol)

    # (c) TABLE I steps 13-15 (mirrored): SVD → B̂^pr (image dim D̂ = D_f)
    B_pr = @timeit timer "CBE step c" _cbe_get_Bpr_r2l(R_orth_tr, AR, D_eff, alg_svd, cbe_tol, safety)
    isnothing(B_pr) && return NaN, true

    # (d) TABLE I steps 16-23 (mirrored): GETCORTH + final SVD → B̂^tr (image dim D̃)
    B_tr = @timeit timer "CBE step d" _cbe_get_Btr_r2l(B_pr, AL, AR, Λ, W1, W2, GL, GR, D_tilde, alg_svd, cbe_tol)

    # Bond expansion: B^ex_{ℓ+1} = B_{ℓ+1} ⊕ B̂^tr_{ℓ+1}, C^{e.i.}_ℓ zero-padded
    ϵp = _cbe_bond_expand_r2l!(ψ, pos, AR, B_tr, project_error)

    return ϵp, false
end

# ────────────────────────────────────────────────────────────────────────────
# Fig. S-2 step (a): GETRORTH
# ────────────────────────────────────────────────────────────────────────────

"""
    _cbe_getrorth_l2r(GR, W2, AR, Λ) → (U, S, Vᴴ)

Fig. S-2(a), TABLE I steps 1-6, L→R sweep. GETRORTH.
Computes R^orth_ℓ = (I - B B†)(Λ · B_{ℓ+1} · W_{ℓ+1} · R_{ℓ+2}), compact SVD (no truncation).
Only U·S flows into GETLORTH (step b); Vᴴ is discarded by the caller.
"""
function _cbe_getrorth_l2r(GR, W2, AR, Λ)
    @plansor opt=true tmp[-1; -4 -3 -2] := Λ[-1; 1] * AR[1 2; 3] * W2[-2 -3; 2 4] * GR[3 4; -4]
    @plansor opt=true tmp2[-1; -4 -3 -2] := tmp[-1; 2 1 -2] * conj(AR[3 1; 2]) * AR[3 -3; -4]
    return svd_compact!(tmp - tmp2)
end

"""
    _cbe_getlorth_r2l(GL, W1, AL, Λ) → (U, S, Vᴴ)

Fig. S-2(a) mirrored, TABLE I steps 1-6, R→L sweep. GETLORTH (mirror).
Returns compact SVD of L^orth; only S·Vᴴ flows into GETRORTH (step b).
"""
function _cbe_getlorth_r2l(GL, W1, AL, Λ)
    @plansor opt=true tmp[-1 -2 -4; -3] := GL[-1 2; 1] * AL[1 3; 4] * W1[2 -2; 3 -4] * Λ[4; -3]
    @plansor opt=true tmp2[-1 -2 -4; -3] := tmp[1 2 -4; -3] * conj(AL[1 2; 3]) * AL[-1 -2; 3]
    return svd_compact!(tmp - tmp2)
end

# ────────────────────────────────────────────────────────────────────────────
# Fig. S-2 step (b): GETLORTH
# ────────────────────────────────────────────────────────────────────────────

"""
    _cbe_getlorth_l2r(GL, W1, AL, U, S, D_prime, alg_svd) → L_orth_tr

Fig. S-2(b), TABLE I steps 7-12, L→R sweep.

Computes L^orth_ℓ by projecting out the kept (A_ℓ) subspace:

    L^imp_ℓ  = GL · W_ℓ · A_ℓ · U · S
    L^orth_ℓ = L^imp_ℓ - A_ℓ · (A†_ℓ · L^imp_ℓ)

Then SVD-truncates center bond D → D'.
"""
function _cbe_getlorth_l2r(GL, W1, AL, U, S, D_prime::Int, alg_svd, cbe_tol::Real)
    @plansor opt=true tmp[-1 -2 -4; -3] := GL[-1 2; 1] * AL[1 3; 4] * W1[2 -2; 3 -4] * U[4; 5] * S[5; -3]
    @plansor opt=true tmp2[-1 -2 -4; -3] := tmp[1 2 -4; -3] * conj(AL[1 2; 3]) * AL[-1 -2; 3]
    U_L, S_L, _, _ = svd_trunc!(tmp - tmp2; trunc=_cbe_preselect_trunc(D_prime, cbe_tol), alg=alg_svd)
    return U_L * S_L
end

"""
    _cbe_getrorth_r2l(GR, W2, AR, S, Vᴴ, D_prime, alg_svd) → R_orth_tr

Fig. S-2(b) mirrored, TABLE I steps 7-12, R→L sweep. Mirror of `_cbe_getlorth_l2r`.
"""
function _cbe_getrorth_r2l(GR, W2, AR, S, Vᴴ, D_prime::Int, alg_svd, cbe_tol::Real)
    @plansor opt=true tmp[-1; -4 -3 -2] := S[-1; 1] * Vᴴ[1; 2] * AR[2 4; 3] * W2[-2 -3; 4 5] * GR[3 5; -4]
    @plansor opt=true tmp2[-1; -4 -3 -2] := tmp[-1; 2 1 -2] * conj(AR[3 1; 2]) * AR[3 -3; -4]
    _, S_R, Vᴴ_R, _ = svd_trunc!(tmp - tmp2; trunc=_cbe_preselect_trunc(D_prime, cbe_tol), alg=alg_svd)
    return S_R*Vᴴ_R
end


"""
    _cbe_get_Apr_l2r(L_orth_tr, AL, D_f, alg_svd) → A_pr

Fig. S-2(c), TABLE I steps 13-15, L→R sweep.

Redirects the MPO leg of L_orth_tr via repartition(L_orth_tr, 2, 2), then SVD
truncates to D_f singular values (D̂ = D_f). Step (b) already projected the
candidate into the kept-space complement; the optional safety step
reorthogonalizes the result if needed.
"""
function _cbe_get_Apr_l2r(L_orth_tr, AL, D_f::Int, alg_svd, cbe_tol::Real, safety::Bool)
    # Step (b) already projected L_orth_tr into the complement of AL before the
    # MPO leg was redirected. Projecting once more after `repartition` is not
    # equivalent to the paper's SVD. Instead, remove numerical null singular
    # vectors and then explicitly verify/reorthogonalize the result.
    X_orth = repartition(L_orth_tr, 2, 2)
    fact = svd_trunc!(X_orth; trunc=_cbe_preselect_trunc(D_f, cbe_tol), alg=alg_svd)
    U_pr, _, _, _ = fact
    safety || return U_pr
    _cbe_Apr_orthogonality(AL, U_pr) <= _cbe_preselect_orth_tol() && return U_pr
    U_pr = _cbe_project_Apr_complement(AL, U_pr)
    fact_re = svd_trunc!(U_pr; trunc=_cbe_preselect_trunc(D_f, cbe_tol), alg=alg_svd)
    U_re, _, _, _ = fact_re
    _cbe_Apr_orthogonality(AL, U_re) <= _cbe_preselect_orth_tol() || return nothing
    return U_re
end

"""
    _cbe_get_Bpr_r2l(R_orth_tr, AR, D_f, alg_svd) → B_pr

Fig. S-2(c) mirrored, TABLE I steps 13-15, R→L sweep. Mirror of `_cbe_get_Apr_l2r`.
"""
function _cbe_get_Bpr_r2l(R_orth_tr, AR, D_f::Int, alg_svd, cbe_tol::Real, safety::Bool)
    # See `_cbe_get_Apr_l2r`: the orthogonal projection has already happened
    # before the MPO/truncation legs are fused into the preselection matrix.
    X_orth = repartition(R_orth_tr, 2, 2)
    fact = svd_trunc!(X_orth; trunc=_cbe_preselect_trunc(D_f, cbe_tol), alg=alg_svd)
    _, _, V_pr, _ = fact
    safety || return V_pr
    _cbe_Bpr_orthogonality(V_pr, AR) <= _cbe_preselect_orth_tol() && return V_pr
    V_pr = _cbe_project_Bpr_complement(V_pr, AR)
    fact_re = svd_trunc!(V_pr; trunc=_cbe_preselect_trunc(D_f, cbe_tol), alg=alg_svd)
    _, _, V_re, _ = fact_re
    _cbe_Bpr_orthogonality(V_re, AR) <= _cbe_preselect_orth_tol() || return nothing
    return V_re
end

function _cbe_Apr_orthogonality(AL, A_pr)
    @plansor opt=true overlap[-1; -2] := conj(AL[1 2; -1]) * A_pr[1 2; -2]
    return norm(overlap)
end

function _cbe_Bpr_orthogonality(B_pr, AR)
    ARr = repartition(AR, 1, 2)
    @plansor opt=true overlap[-1; -2] := B_pr[-1; 1 2] * conj(ARr[-2; 1 2])
    return norm(overlap)
end

function _cbe_project_Apr_complement(AL, A_pr)
    @plansor opt=true overlap[-1; -2] := conj(AL[1 2; -1]) * A_pr[1 2; -2]
    @plansor opt=true proj[-1 -2; -3] := AL[-1 -2; 1] * overlap[1; -3]
    return A_pr - proj
end

function _cbe_project_Bpr_complement(B_pr, AR)
    ARr = repartition(AR, 1, 2)
    @plansor opt=true overlap[-1; -2] := B_pr[-1; 1 2] * conj(ARr[-2; 1 2])
    @plansor opt=true proj[-1; -2 -3] := overlap[-1; 1] * ARr[1; -2 -3]
    return B_pr - proj
end

# ────────────────────────────────────────────────────────────────────────────
# Fig. S-2 step (d): GETCORTH + final SVD → Â^tr
# ────────────────────────────────────────────────────────────────────────────

"""
    _cbe_get_Atr_l2r(A_pr, AL, AR, Λ, W1, W2, GL, GR, D_tilde, alg_svd) → A_tr

Fig. S-2(d), TABLE I steps 16-23, L→R sweep.

TABLE I steps 16-21 (GETCORTH): with the MPO bond now closed, compute
    C^orth_{ℓ+1} = Â^pr† · H^{2s}_{ℓ} |ψ^{2s}_{ℓ}⟩ · B̄†_{ℓ+1}
at 1-site cost (Â^pr has image dim D̂ ≈ D, no extra d factor).

TABLE I steps 22-23: SVD-truncate C^orth_{ℓ+1} to D̃ singular values,
giving ũ; return Â^tr = Â^pr · ũ with codomain=(D_left, phys), image dimension D̃.
"""
function _cbe_get_Atr_l2r(A_pr, AL, AR, Λ, W1, W2, GL, GR, D_tilde::Int, alg_svd, cbe_tol::Real)
    @plansor opt=true L_pr[-1; -2 -3] := GL[4 3; 1] * AL[1 2; -2] * W1[3 5; 2 -3] * conj(A_pr[4 5; -1])
    @plansor opt=true tmp[-1; -3 -2] := L_pr[-1; 1 6] * Λ[1; 2] * AR[2 4; 3] * W2[6 -2; 4 5] * GR[3 5; -3]
    @plansor opt=true tmp2[-1; -3 -2] := tmp[-1; 2 1] * conj(AR[3 1; 2]) * AR[3 -2; -3]
    tilde_u, _, _, _ = svd_trunc!(tmp - tmp2; trunc=_cbe_preselect_trunc(D_tilde, cbe_tol), alg=alg_svd)
    @plansor A_tr[-1 -2; -3] := A_pr[-1 -2; 1] * tilde_u[1; -3]
    return A_tr
end

"""
    _cbe_get_Btr_r2l(B_pr, AL, AR, Λ, W1, W2, GL, GR, D_tilde, alg_svd) → B_tr

Fig. S-2(d) mirrored, TABLE I steps 16-23, R→L sweep.

Mirror of `_cbe_get_Atr_l2r`.
"""
function _cbe_get_Btr_r2l(B_pr, AL, AR, Λ, W1, W2, GL, GR, D_tilde::Int, alg_svd, cbe_tol::Real)
    @plansor opt=true R_pr[-1 -2; -3] := AR[-1 1; 2] * W2[-2 4;1 3] * GR[2 3; 5] * conj(B_pr[-3; 5 4])
    @plansor opt=true tmp[-1 -2; -3] := GL[-1 2; 1] * AL[1 3; 4] * W1[2 -2; 3 6] * Λ[4; 5] * R_pr[5 6; -3]
    @plansor opt=true tmp2[-1 -2; -3] := tmp[1 2; -3] * conj(AL[1 2; 3]) * AL[-1 -2; 3]
    _, _, tilde_vh, _ = svd_trunc!(tmp - tmp2; trunc=_cbe_preselect_trunc(D_tilde, cbe_tol), alg=alg_svd)
    @plansor B_tr[-1; -3 -2] := tilde_vh[-1; 1] * B_pr[1; -3 -2]
    return B_tr
end

# ────────────────────────────────────────────────────────────────────────────
# Bond expansion: A^ex_ℓ = A_ℓ ⊕ Â^tr_ℓ, C^{e.i.} zero-padded
# ────────────────────────────────────────────────────────────────────────────

"""
    _cbe_bond_expand_l2r!(ψ, pos, AL, A_tr, project_error)

Expand bond (pos, pos+1) in the L→R direction (Gleis et al. Algorithm 1 step ii).

    A^ex_ℓ          = catdomain(AL, A_tr[1,1,1])                    → ψ.AL[pos]
    C^{e.i.}_{ℓ+1}  = (A^ex)† · AL · ψ.AC[pos+1]  (zero-padded by D̃) → ψ.AC[pos+1]

Since A_tr ⊥ AL, the state is numerically unchanged: A^ex · C^{e.i.} = AL · C_{ℓ+1}.
Returns the explicit projection error if `project_error` is true; otherwise `NaN`.
"""
function _cbe_bond_expand_l2r!(ψ, pos::Int, AL, A_tr, project_error::Bool)
    if project_error
        @plansor opt=true old_two_site[-1 -2 -3; -4] := AL[-1 -2; 1] * ψ.AC[pos + 1][1 -3; -4]
    end

    A_tr_inner = A_tr[1, 1, 1]
    A_ex0 = catdomain(AL, A_tr_inner)
    @plansor opt=true new_ac0[-1 -2; -3] := conj(A_ex0[1 2; -1]) * AL[1 2; 3] * ψ.AC[pos+1][3 -2; -3]

    # FiniteMPS performs a final no-truncation orthogonalization after oplus.
    # This keeps the enlarged basis canonical while preserving the state.
    A_ex, R = left_orth(A_ex0)
    @plansor opt=true new_ac[-1 -2; -3] := R[-1; 1] * new_ac0[1 -2; -3]

    ψ.AC[pos] = (A_ex, R)
    ψ.AC[pos + 1] = new_ac
    if project_error
        @plansor opt=true new_two_site[-1 -2 -3; -4] := A_ex[-1 -2; 1] * new_ac[1 -3; -4]
        return norm(old_two_site - new_two_site)
    end
    return NaN
end

"""
    _cbe_bond_expand_r2l!(ψ, pos, AR, B_tr, project_error)

Expand bond (pos, pos+1) in the R→L direction (Gleis et al. Algorithm 1 step ii).

    B^ex_{ℓ+1}   = catcodomain(AR, B_tr[1,1,1])               → ψ.AR[pos+1]
    C^{e.i.}_ℓ   = ψ.AC[pos] · AR · B^ex†  (zero-padded by D̃) → ψ.AC[pos]

Since B_tr ⊥ AR, the state is numerically unchanged: C^{e.i.} · B^ex = C_ℓ · AR.
Returns the explicit projection error if `project_error` is true; otherwise `NaN`.
"""
function _cbe_bond_expand_r2l!(ψ, pos::Int, AR, B_tr, project_error::Bool)
    if project_error
        @plansor opt=true old_two_site[-1 -2 -3; -4] := ψ.AC[pos][-1 -2; 1] * AR[1 -3; -4]
    end

    B_tr_inner = B_tr[1, 1, 1]
    B_ex0 = repartition(catcodomain(repartition(AR, 1, 2), B_tr_inner), 2, 1)
    @plansor opt=true new_ac0[-1 -2; -3] := ψ.AC[pos][-1 -2; 1] * AR[1 2; 3] * conj(B_ex0[-3 2; 3])

    # Mirror of the final no-truncation orthogonalization after oplus.
    L, B_ex_tail = right_orth(_transpose_tail(B_ex0))
    B_ex = _transpose_front(B_ex_tail)
    @plansor opt=true new_ac[-1 -2; -3] := new_ac0[-1 -2; 1] * L[1; -3]

    ψ.AC[pos + 1] = (L, B_ex)
    ψ.AC[pos] = new_ac
    if project_error
        @plansor opt=true new_two_site[-1 -2 -3; -4] := new_ac[-1 -2; 1] * B_ex[1 -3; -4]
        return norm(old_two_site - new_two_site)
    end
    return NaN
end
