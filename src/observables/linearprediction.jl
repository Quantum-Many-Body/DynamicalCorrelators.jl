# Linear-prediction extrapolation of finite-time correlation data.
# Implements Sec. 7.4 "Green's functions 2: Linear prediction" (Eqs. (178)-(185))
# of Paeckel et al., "Time-evolution methods for matrix-product states",
# Ann. Phys. 411, 167998 (2019).
#
# A time series built from a sum of damped oscillations,
# x_n = Σⱼ cⱼ αⱼⁿ with αⱼ = exp(-γⱼ Δt - i ωⱼ Δt), satisfies exactly an order-p
# recursion x̃ₙ = -∑ᵢ aᵢ x_{n-i}. Fitting the coefficients aᵢ on a window of the
# MPS data and rolling the recursion forward extends the series to arbitrary
# times, so the subsequent Fourier transform needs only minimal damping.

using LinearAlgebra: Hermitian, eigvals, cholesky, issuccess, pinv

"""
    LinearPrediction(; p, D=p+1, N=nothing, epsilon=0.0, clean_poles=true, pole_tol=0.0)

Algorithm settings for linear-prediction extrapolation of a finite time series
(Sec. 7.4, Eqs. (178)-(185) of Paeckel et al., Ann. Phys. 411, 167998 (2019)).

The series `xₙ` sampled at `tₙ = n·Δt` is modeled as a sum of damped
oscillations `xₙ = Σⱼ cⱼ αⱼⁿ` with `αⱼ = exp(-γⱼ Δt - i ωⱼ Δt)`, which
satisfies exactly an order-`p` recursion

```math
\\tilde{x}_n = -\\sum_{i=1}^{p} a_i x_{n-i} .
```

The coefficients are fitted by minimizing `ε = Σₖ |x̃ₖ - xₖ|²` over a fitting
window, giving the normal equations `a = -R⁻¹ r` with
`Rᵢⱼ = Σₖ x*ₖ₋ᵢ xₖ₋ⱼ`, `rᵢ = Σₖ x*ₖ₋ᵢ xₖ` (Eqs. (179)-(182)).

# Keywords
- `p::Integer`: number of recursion coefficients, i.e. the assumed number of
  poles. Choose `N ≳ 2p` so the least-squares problem is well overdetermined.
- `D::Integer = p+1`: number of initial data points *discarded* before fitting
  (the review uses 0-based sample indices; discarding `D` points means the
  fitting window starts at `x[D+1]` in Julia indexing). The early-time
  transient is usually not described by a few poles, so `D` should remove it.
  Must satisfy `D ≥ p`.
- `N::Union{Nothing,Integer} = nothing`: length of the fitting window, samples
  `x[D+1 : D+N]`. `nothing` uses all remaining data; choosing a smaller `N`
  reserves the window `x[D+N+1 : end]` for validating the prediction, as
  recommended in the review (see [`validation_error`](@ref)).
- `epsilon::Real = 0.0`: diagonal shift `R → R + ε·I`, the remedy for a
  (nearly) singular `R` suggested in the review. With `epsilon = 0`, a singular
  `R` falls back to its pseudoinverse with a warning (reducing `p` is the
  alternative remedy).
- `clean_poles::Bool = true`: remove unphysical poles. The eigenvalues `αᵢ` of
  the companion matrix (Eqs. (183)-(185)) with `|αᵢ| > 1 + pole_tol` describe
  exponentially growing modes; per the reference cited in the review, setting
  those contributions to zero works best. The recursion is then rebuilt from
  the remaining stable poles only and its order drops accordingly.
- `pole_tol::Real = 0.0`: tolerance when deciding whether a pole is growing.
  Increase it to keep marginally growing poles, e.g. undamped modes pushed
  slightly above `|α| = 1` by fitting noise.
"""
struct LinearPrediction
    p::Int
    D::Int
    N::Union{Nothing, Int}
    epsilon::Float64
    clean_poles::Bool
    pole_tol::Float64
    function LinearPrediction(; p::Integer, D::Integer=p+1, N::Union{Nothing,Integer}=nothing,
                              epsilon::Real=0.0, clean_poles::Bool=true, pole_tol::Real=0.0)
        p >= 1 || throw(ArgumentError("p must be a positive integer, got $p"))
        D >= p || throw(ArgumentError("D ($D) must be at least p ($p): evaluating x̃ₖ needs x_{k-p} inside the computed data"))
        isnothing(N) || N >= 1 || throw(ArgumentError("N must be a positive integer, got $N"))
        epsilon >= 0 || throw(ArgumentError("epsilon must be non-negative, got $epsilon"))
        return new(p, D, N, epsilon, clean_poles, pole_tol)
    end
end

"""
    LinearPredictionModel

Fitted linear-prediction model returned by [`fit_linear_prediction`](@ref).

# Fields
- `a`: effective recursion coefficients, `x̃ₙ = -Σᵢ aᵢ x_{n-i}`. Length `p`
  normally, or smaller when pole cleaning removed growing modes (see
  [`LinearPrediction`](@ref)).
- `a_raw`: the `p` coefficients of the raw least-squares fit, before cleaning.
- `poles`: eigenvalues `αᵢ` of the companion matrix (Eqs. (183)-(185)), with
  `αᵢ = exp(-γᵢ Δt - i ωᵢ Δt)`. See [`pole_analysis`](@ref).
- `kept`: mask marking which poles survived the cleaning.
- `rms`: root-mean-square fitting error `sqrt(ε/N)` on the fitting window.
- `D`, `N`, `p`: the window parameters used by the fit.
"""
struct LinearPredictionModel{A<:AbstractVector, B<:AbstractVector, P<:AbstractVector}
    a::A
    a_raw::B
    poles::P
    kept::Vector{Bool}
    rms::Float64
    D::Int
    N::Int
    p::Int
end

"""
    fit_linear_prediction(x, alg::LinearPrediction) -> LinearPredictionModel

Fit the recursion coefficients `aᵢ` of Eq. (178) on the time series `x` by
solving the normal equations (180)-(182) over the fitting window
`x[alg.D+1 : alg.D+alg.N]`, then build the companion matrix (185), extract its
eigenvalues, and (if `alg.clean_poles`) remove the growing `|α| > 1` poles.
"""
function fit_linear_prediction(x::AbstractVector, alg::LinearPrediction)
    n = length(x)
    p, D = alg.p, alg.D
    N = isnothing(alg.N) ? n - D : alg.N
    N >= 1 || throw(ArgumentError("empty fitting window: D = $D leaves no data in a series of length $n"))
    D + N <= n || throw(ArgumentError("fitting window x[$(D+1):$(D+N)] exceeds the data length $n; reduce N or D"))
    N < p && @warn "fitting window N = $N is smaller than the number of coefficients p = $p; the normal-equation matrix R will be rank-deficient"
    T = promote_type(Float64, eltype(x))
    R = zeros(T, p, p)
    r = zeros(T, p)
    @inbounds for k in (D+1):(D+N)
        for i in 1:p
            xi = conj(x[k-i])
            r[i] += xi*x[k]
            for j in 1:p
                R[i, j] += xi*x[k-j]
            end
        end
    end
    a_raw = _solve_normal_equation(Hermitian(R), r, alg.epsilon)
    rms = _fitting_rms(x, a_raw, D, N)
    poles = _companion_poles(a_raw)
    kept, a = _clean_poles(a_raw, poles, alg.clean_poles, alg.pole_tol)
    return LinearPredictionModel(a, a_raw, poles, kept, rms, D, N, p)
end

# Solve (R + εI) a = -r. R is Hermitian positive semidefinite by construction;
# a Cholesky factorization is used and a singular/indefinite R falls back to
# the pseudoinverse, cf. the "numerical pitfalls" paragraph of the review.
function _solve_normal_equation(R::Hermitian, r::AbstractVector, epsilon::Real)
    M = epsilon > 0 ? Hermitian(Matrix(R) + epsilon*I) : R
    F = cholesky(M; check=false)
    issuccess(F) && return -(F \ r)
    @warn "the normal-equation matrix R is singular; using its pseudoinverse. Consider epsilon > 0 or a smaller p (the remedies suggested in the review)."
    return -(pinv(Matrix(M)) * r)
end

function _fitting_rms(x::AbstractVector, a::AbstractVector, D::Integer, N::Integer)
    p = length(a)
    S = promote_type(eltype(x), eltype(a))
    err = zero(real(S))
    @inbounds for k in (D+1):(D+N)
        pred = zero(S)
        for i in 1:p
            pred -= a[i]*x[k-i]
        end
        err += abs2(pred - x[k])
    end
    return sqrt(err/N)
end

# Eigenvalues αᵢ of the companion matrix A (Eqs. (183)-(185)).
function _companion_poles(a::AbstractVector)
    p = length(a)
    CT = complex(eltype(a))
    p == 1 && return CT[-a[1]]
    A = zeros(CT, p, p)
    @inbounds for j in 1:p
        A[1, j] = -a[j]
    end
    @inbounds for i in 2:p
        A[i, i-1] = one(CT)
    end
    return eigvals(A)
end

# Zero out the contributions of growing poles (|α| > 1 + pole_tol) by rebuilding
# the recursion from the stable poles only: if ∏(z - αⱼ) = z^m + Σᵢ bᵢ z^{m-i},
# the surviving signal satisfies xₙ = -Σᵢ bᵢ x_{n-i}.
function _clean_poles(a_raw::AbstractVector, poles::AbstractVector, clean::Bool, pole_tol::Real)
    p = length(a_raw)
    clean || return trues(p), a_raw
    kept = [abs(α) <= 1 + pole_tol for α in poles]
    all(kept) && return kept, a_raw
    any(kept) || throw(ArgumentError("every pole has |α| > 1 + pole_tol = $(1 + pole_tol); the fitted recursion is fully unstable. Try a different fitting window, a smaller p, or clean_poles=false."))
    a = _poly_coeffs_from_roots(poles[kept])
    eltype(a_raw) <: Real && (a = _real_if_close(a))
    return kept, a
end

# Coefficients b₁..bₘ of the monic polynomial ∏(z - αⱼ) = z^m + b₁ z^{m-1} + ... + bₘ.
function _poly_coeffs_from_roots(roots::AbstractVector)
    b = [one(eltype(roots))]
    for α in roots
        bnew = Vector{eltype(roots)}(undef, length(b)+1)
        bnew[1] = b[1]
        @inbounds for i in 2:length(b)
            bnew[i] = b[i] - α*b[i-1]
        end
        bnew[end] = -α*b[end]
        b = bnew
    end
    return b[2:end]
end

# Drop negligible imaginary parts left over from pairing complex-conjugate roots.
function _real_if_close(v::AbstractVector{<:Complex}; tol::Real=1e-12)
    scale = maximum(abs, v)
    return all(x -> abs(imag(x)) <= tol*max(one(scale), scale), v) ? real.(v) : v
end
_real_if_close(v::AbstractVector{<:Real}; tol::Real=1e-12) = v

# Roll the recursion x̃ₙ = -Σᵢ aᵢ x_{n-i} forward `nsteps` times from `seed`
# (the m most recent samples, in chronological order).
function _roll(a::AbstractVector, seed::AbstractVector, nsteps::Integer)
    m = length(a)
    S = promote_type(eltype(a), eltype(seed))
    buf = Vector{S}(undef, m + nsteps)
    copyto!(buf, 1, seed, 1, m)
    @inbounds for s in 1:nsteps
        acc = zero(S)
        base = m + s
        for i in 1:m
            acc -= a[i]*buf[base-i]
        end
        buf[base] = acc
    end
    return buf[(m+1):end]
end

"""
    extend_series(x, model::LinearPredictionModel, nsteps::Integer) -> Vector

Extend the time series `x` by `nsteps` predicted samples using the fitted
recursion, seeded with the last `length(model.a)` data points. The returned
vector is `[x; predictions]`; the matching time grid is
`range(ts[1], step=step(ts), length=length(ts)+nsteps)`.
"""
function extend_series(x::AbstractVector, model::LinearPredictionModel, nsteps::Integer)
    nsteps >= 0 || throw(ArgumentError("nsteps must be non-negative, got $nsteps"))
    m = length(model.a)
    n = length(x)
    n >= m || throw(ArgumentError("need at least $(model.p) data points to seed the recursion, got $n"))
    return vcat(x, _roll(model.a, view(x, (n-m+1):n), nsteps))
end

"""
    validation_error(model::LinearPredictionModel, x) -> NamedTuple

Roll the fitted recursion over the validation window `x[D+N+1 : end]` — data
that was *not* used for the fit — seeded with the last samples of the fitting
window, and compare with the computed data. Returns
`(rms, prediction, window)`: the root-mean-square error, the predicted samples,
and the index range of the validation window.
"""
function validation_error(model::LinearPredictionModel, x::AbstractVector)
    D, N = model.D, model.N
    n = length(x)
    first = D + N + 1
    nval = n - first + 1
    nval > 0 || throw(ArgumentError("no validation data: the fitting window already ends at the last sample (D + N = $(D+N) = length(x)); choose a smaller N to reserve a validation window"))
    m = length(model.a)
    m <= D + N || throw(ArgumentError("recursion order m = $m exceeds the $(D+N) samples before the validation window"))
    pred = _roll(model.a, view(x, (D+N-m+1):(D+N)), nval)
    rms = sqrt(sum(abs2, pred .- view(x, first:n))/nval)
    return (rms=rms, prediction=pred, window=first:n)
end

"""
    pole_analysis(model::LinearPredictionModel, dt=1.0) -> NamedTuple

Interpret the companion-matrix eigenvalues `αᵢ = exp(-γᵢ Δt - i ωᵢ Δt)` of the
fitted model (Eqs. (183)-(185)) as frequencies and damping rates, where `dt` is
the time step of the sampled series. Returns `(poles, frequency, damping,
kept)` with `frequency = -arg(αᵢ)/dt` and `damping = -log|αᵢ|/dt`; `kept` marks
the poles that survived pole cleaning.
"""
function pole_analysis(model::LinearPredictionModel, dt::Real=1.0)
    α = model.poles
    return (poles=α,
            frequency=-angle.(α)/dt,
            damping=-log.(abs.(α))/dt,
            kept=model.kept)
end

"""
    extrapolate_series(x::AbstractVector, alg::LinearPrediction, nsteps) -> (Vector, LinearPredictionModel)

Fit [`fit_linear_prediction`](@ref) on `x` and [`extend_series`](@ref) it by
`nsteps` samples. Returns the extended series and the fitted model.

    extrapolate_series(gf::AbstractArray, alg::LinearPrediction, nsteps) -> (Array, Array{LinearPredictionModel})

Array version with time on the last axis: an independent model is fitted to
every matrix element of `gf` (e.g. every `Gᵢⱼ(t)` of a `dcorrelator` result, or
every `G_{xy}(k, t)` after `fourier_kt`), and each element is extended by
`nsteps` samples. Returns the extended array and the per-element models.

# Example
```julia
alg = LinearPrediction(p=12, D=13, N=150)   # keep a validation window
gf_ext, models = extrapolate_series(gf_rt, alg, 400)
ts_ext = range(ts[1], step=step(ts), length=size(gf_ext, 3))
gf_rw = fourier_rw(gf_ext, ts_ext, ws; broadentype=(0.01, "G"))
```
"""
function extrapolate_series(x::AbstractVector, alg::LinearPrediction, nsteps::Integer)
    model = fit_linear_prediction(x, alg)
    return extend_series(x, model, nsteps), model
end

function extrapolate_series(gf::AbstractArray, alg::LinearPrediction, nsteps::Integer)
    nd = ndims(gf)
    nd >= 2 || throw(ArgumentError("expected an array whose last axis is time"))
    nt = size(gf, nd)
    leads = CartesianIndices(axes(gf)[1:(nd-1)])
    models = map(I -> fit_linear_prediction(view(gf, I, :), alg), leads)
    S = promote_type(eltype(gf), mapreduce(m -> eltype(m.a), promote_type, models))
    out = Array{S}(undef, size(gf)[1:(nd-1)]..., nt + nsteps)
    for I in leads
        out[I, :] .= extend_series(view(gf, I, :), models[I], nsteps)
    end
    return out, models
end
