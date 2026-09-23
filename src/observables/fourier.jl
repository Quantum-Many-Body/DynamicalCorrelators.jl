# Fourier transform utilities for correlation functions.
# Nyquist constraints: t = 0:Δt:T, ω_max < π/Δt, Δω < 2π/T (Δω ≈ π/2T)

"""
    broaden_gauss(t, eta)

Gaussian broadening (damping) function: `exp(-(η·t)²)`.
Produces Gaussian peaks in frequency space with width proportional to `η`.
"""
function broaden_gauss(t::Real, eta::Real)
    return exp(-(eta*t)^2)
end

"""
    broaden_gauss_normalized(t, eta, tmax)

Window-normalized Gaussian broadening (damping) function: `exp(-η·(t/tmax)^2)`.

The damping at the edge of the time window is `exp(-η)` regardless of `tmax`,
so the whole time window is used and the frequency resolution `~ sqrt(2η)/tmax`
keeps improving as the time window grows (unlike `"G"`, whose resolution is
pinned at `~ sqrt(2)·η`). Typical choice: `η ≈ 2 ~ 4`.
"""
function broaden_gauss_normalized(t::Real, eta::Real, tmax::Real)
    return exp(-eta*(t/tmax)^2)
end

"""
    broaden_lorentz(t, eta)

Lorentzian broadening (damping) function: `exp(-η|t|)`.
Produces Lorentzian peaks in frequency space with half-width `η`.
"""
function broaden_lorentz(t::Real, eta::Real)
    return exp(-eta*abs(t))
end

"""
    blackman_window(t, T)

Blackman window function for spectral analysis.
Provides good sidelobe suppression: `0.42 - 0.5cos(2πt/T) + 0.08cos(4πt/T)`.
"""
function blackman_window(t::Real, T::Real)
    return 0.42 - 0.5*cos(2π*t/T) + 0.08*cos(4π*t/T)
end

"""
    parzen_window(t, T)

Parzen (de la Vallée Poussin) window function.
A piecewise cubic window that smoothly tapers to zero at `|t/T| = 1`.
"""
function parzen_window(t::Real, T::Real)
	if abs(t/T) <= 1/2
		return 6*abs(t/T)^3 - 6*(t/T)^2 + 1
	elseif abs(t/T) <= 1
		return 2*(1 - abs(t/T))^3
	else
		return 0.0
	end
end

"""
    damping(t, broadentype)

Apply a broadening/damping function at time `t`.

`broadentype` is a tuple `(parameter, type_string)` where:
- `type_string` is one of `"G"` (Gaussian), `"GN"` (window-normalized Gaussian),
  `"L"` (Lorentzian), `"B"` (Blackman), `"P"` (Parzen).
- `parameter` is `η` for Gaussian/Lorentzian or `T` for window functions.

For `"GN"`, keyword `tmax` (the end of the time window) is required; see
[`broaden_gauss_normalized`](@ref).
"""
function damping(t, broadentype; tmax::Union{Nothing,Real}=nothing)
    if broadentype[2] == "G"
        return broaden_gauss(t, broadentype[1])
    elseif broadentype[2] == "GN"
        isnothing(tmax) && throw(ArgumentError("broadentype \"GN\" requires keyword `tmax` (end of the time window)."))
        return broaden_gauss_normalized(t, broadentype[1], tmax)
    elseif broadentype[2] == "L"
        return broaden_lorentz(t, broadentype[1])
    elseif broadentype[2] == "B"
        return blackman_window(t, broadentype[1])
    elseif broadentype[2] == "P"
        return parzen_window(t, broadentype[1])
    else
        throw(ArgumentError("Invalid broadening type: $broadentype"))
    end
end

"""
    _spatial_phase(kernel, k, Δr)

Phase factor of the spatial Fourier kernel for separation `Δr`: the full
`exp(-i k·Δr)` for `kernel === :exp`, or its inversion-symmetric part
`cos(k·Δr)` for `kernel === :cos`.
"""
_spatial_phase(kernel::Symbol, k::AbstractArray{<:Number}, Δr) = kernel === :cos ? cos(dot(k, Δr)) : cis(-dot(k, Δr))

"""
    fourier_kt(gf_rt, rs, k; regroup=[collect(1:size(gf_rt,1))], center, kernel=:exp)

Spatial Fourier transform of a real-space Green's function `G(r, t)` at a single
momentum point `k`.

MPS sites do not necessarily correspond one-to-one to physical lattice sites —
one MPS site may be a single orbital of a multi-orbital unit cell, one leg of a
ladder, etc. `regroup` therefore partitions the MPS site indices into physical
channels and the result is a *matrix over groups*:

```math
\\mathrm{dest}[x, y, t] = \\sum_{j \\in \\mathrm{regroup}[x]}\\ \\sum_{i \\in \\mathrm{regroup}[y]}
G_{ji}(t)\\, e^{-i\\, k \\cdot (r_i - r_j)} .
```

The sum inside each group pair is *coherent*, with the phase computed from each
MPS site's true position `rs` (so intra-cell orbital offsets are included), and
no normalization such as `1/N_cells` is applied. Axis 1 of `gf_rt` is the
measurement site and axis 2 the source site, matching the output of
`dcorrelator`.

# Arguments
- `gf_rt`: Green's function array of shape `(N_mps, N_mps, N_times)` (see
  `center` for the case where only a subset of source sites was evolved).
- `rs`: position vector of every MPS site, `length(rs) == size(gf_rt, 1)`.
- `k`: momentum vector of the same dimension as the entries of `rs`.
- `regroup`: groups of MPS site indices. Should be a partition of `1:N_mps`:
  sites absent from every group are dropped from the sum, sites appearing in
  several groups are counted multiple times.
- `center`: required keyword, normally `nothing`. If `gf_rt` was computed only
  for a subset of source sites (e.g. from `dcorrelator(gs, H, op, indices)`),
  pass the corresponding physical site indices (`id <= L ? id : id - L` for
  greater/lesser ids). Then `size(gf_rt, 2) == length(center)`, only axis 1 is
  regrouped, the output has size `(length(regroup), length(center), N_times)`,
  and the phase is `exp(-i k·(rs[center[y]] - rs[j]))`.
- `kernel`: `:exp` (default) for the full complex phase `exp(-i k·Δr)`, or
  `:cos` to keep only its inversion-symmetric part `cos(k·Δr)`. For a
  translation-invariant system with spatial inversion symmetry one has
  `G(-x, t) = G(x, t)`, so the sine part of the phase cancels pairwise and
  `:cos` is exact; on numerical data it additionally projects out the small
  inversion-antisymmetric component, which would otherwise produce a
  nonphysical imaginary part in `G(k, ω)`. Use `:cos` with `center` data
  measured from a source at an inversion center, or with a symmetric
  correlator matrix `G_{ij} = G_{ji}`.

# Choosing `regroup`
- **One MPS site = one physical site** (single-orbital chain, snake-ordered 2D
  lattice, ...): keep the default `[collect(1:N_mps)]` — all sites in a single
  group, output is the scalar `G(k, t)` in a `1 × 1` matrix.
- **Orbital/band resolution** (multi-orbital unit cells, bilayers, ladders):
  collect all MPS sites belonging to the *same orbital/leg across all unit
  cells* into one group; the output is then the orbital matrix `G_{αβ}(k, t)`
  (diagonalize it for bonding/antibonding bands). Examples for two orbitals and
  `Nc` unit cells: orbital-major MPS ordering (all orbital-1 sites first)
  `regroup = [collect(1:Nc), collect(Nc+1:2Nc)]`; cell-major ordering (orbitals
  of one cell adjacent) `regroup = [collect(1:2:2Nc), collect(2:2:2Nc)]`.
- **Coarse graining**: to fold several MPS sites into one physical unit while
  keeping the unit-to-unit matrix, group by unit, e.g.
  `regroup = [[2i - 1, 2i] for i in 1:Nc]` gives an `Nc × Nc` output whose
  intra-unit structure is summed with phases. Note this is not yet a pure
  momentum function — sum or trace over the unit indices afterwards if a
  scalar spectrum is needed.
"""
function fourier_kt(gf_rt::AbstractArray, rs::AbstractArray{<:AbstractArray}, k::AbstractArray{<:Number}; regroup::AbstractArray{<:AbstractArray}=[Vector(1:size(gf_rt,1)),], center, kernel::Symbol=:exp)
    kernel in (:exp, :cos) || throw(ArgumentError("Invalid kernel: $kernel (expected :exp or :cos)"))
    if isnothing(center)
        dest = zeros(ComplexF64, length(regroup), length(regroup), size(gf_rt, 3))
        for x in eachindex(regroup), y in eachindex(regroup), l in axes(gf_rt, 3)
            for j in eachindex(regroup[x]), i in eachindex(regroup[y])
                dest[x, y, l] += gf_rt[regroup[x][j], regroup[y][i], l]*_spatial_phase(kernel, k, rs[regroup[y][i]]-rs[regroup[x][j]])
            end
        end
        return dest
    else
        @assert size(gf_rt, 2) == length(center) "Invalid length of center and size(gf_rt, 2)!"
        dest = zeros(ComplexF64, length(regroup), length(center), size(gf_rt, 3))
        for x in eachindex(regroup), y in eachindex(center), l in axes(gf_rt, 3)
            for j in eachindex(regroup[x])
                dest[x, y, l] += gf_rt[regroup[x][j], y, l]*_spatial_phase(kernel, k, rs[center[y]]-rs[regroup[x][j]])
            end
        end
        return dest
    end
end

"""
    fourier_kw(gf_kt, ts, w, dampings; symmetrize=false)

Time Fourier transform of `G(k, t)` to `G(k, ω)` at a single frequency `w`,
with pre-computed damping factors.

With `symmetrize=false` (default) this is the one-sided integral
`∫₀ᵀ dt f(t) e^{iωt} G(k, t)`, whose imaginary part is generically nonzero on
finite-time data. With `symmetrize=±1`, the conjugation symmetry
`G_{yx}(k, -t) = ±conj(G_{xy}(k, t))` is used to fold the two-sided integral
onto the data window:

```math
G_{xy}(k, \\omega) = \\int_0^T dt\\, f(t) \\big[ e^{i\\omega t} G_{xy}(k, t) \\pm e^{-i\\omega t} G_{yx}(k, t)^* \\big] .
```

The sign must match how the correlator was assembled:

- `symmetrize=true` (i.e. `+1`): bare `⟨Aₓ(t) A†ᵧ(0)⟩` correlators without any
  retarded prefactor, which obey `G(-t) = +conj(G(t))`. The result is
  manifestly Hermitian in `(x, y)` at every `ω` and real on the diagonal.
- `symmetrize=-1`: Green's functions carrying the retarded `-im` prefactor,
  which obey `G(-t) = -conj(G(t))`. This is the right choice for the output
  of [`dcorrelator`](@ref), for both spin and electron operators. The result
  is manifestly anti-Hermitian with a purely imaginary diagonal, and the
  spectral function is `-imag(diag)/2` (relative to the usual
  `-imag(one-sided)` convention).

When `gf_kt` is square (all source sites evolved), the conjugated partner
is the transposed element `G_{yx}(k, t)` and the folding is exact. When
`gf_kt` is rectangular (from [`fourier_kt`](@ref) with `center`), the
transposed element was never computed, so each element is folded with its own
conjugate, `G_c(k, -t) = ±conj(G_c(k, t))`; this relies on translation
invariance of the displacement data around each source center, which is the
same assumption under which the folded kernel is derived.

In both cases the result is the full two-sided transform (twice the
corresponding quadrature of the one-sided one), and the unphysical
truncation-induced component in the complementary quadrature is removed.
`ts` must start at `t = 0`.
"""
function fourier_kw(gf_kt::AbstractArray, ts::AbstractRange, w::Number, dampings::AbstractArray; symmetrize::Union{Bool,Integer}=false)
    s = _sym_sign(symmetrize)
    (s != 0 && !iszero(first(ts))) && throw(ArgumentError("`symmetrize` requires time data starting at t = 0, got ts[1] = $(first(ts))."))
    transpose = size(gf_kt, 1) == size(gf_kt, 2)
    dest = zeros(ComplexF64, size(gf_kt, 1), size(gf_kt, 2))
    for x in axes(gf_kt, 1), y in axes(gf_kt, 2)
        temp = _time_integrand(gf_kt, x, y, w, ts, dampings, s, transpose)
        dest[x, y] = integrate(ts, temp)
    end
    return dest
end

"""
    fourier_kw(gf_rt, rs, ts, ks, ws; mthreads=nthreads(), broadentype=(0.05, "G"), regroup=..., kernel=:exp, symmetrize=false)

Full double Fourier transform from `G(r, t)` to `G(k, ω)` over arrays of
momenta `ks` and frequencies `ws`. Multi-threaded over frequencies.

# Arguments
- `gf_rt`: Green's function `(N_mps, N_mps, N_times)` (axis 1 = measurement
  site, axis 2 = source site), or `(N_mps, length(center), N_times)` when
  `center` is given.
- `rs`: site position vectors.
- `ts`: time range.
- `ks`: array of momentum vectors.
- `ws`: array of frequencies.
- `mthreads`: number of threads (default: all available).
- `broadentype`: broadening specification, e.g., `(0.05, "G")` for Gaussian with η=0.05,
  or `(η, "GN")` for the window-normalized Gaussian `exp(-η·(t/tmax)^2)` whose
  resolution improves as the time window grows (typical η ≈ 2 ~ 4).
- `regroup`: groups of MPS site indices defining the physical channels of the
  transform, forwarded to [`fourier_kt`](@ref) — see its docstring for the
  grouping rules and recipes (single group for 1:1 site mappings, one group per
  orbital/leg across all unit cells for band resolution, one group per unit
  for coarse graining).
- `center`: `nothing` (default) or the physical source-site indices when
  `gf_rt` was computed only for a subset of sources; forwarded to
  [`fourier_kt`](@ref).
- `kernel`: forwarded to [`fourier_kt`](@ref); `:exp` (default) or `:cos` to
  keep only the inversion-symmetric `cos(k·Δr)` part of the spatial phase.
- `symmetrize`: `false` (default), `true`/`+1`, or `-1`. If set, fold the
  two-sided time integral onto the data window using
  `G_{ba}(-t) = ±conj(G_{ab}(t))`; requires `ts` to start at `t = 0`. Use
  `-1` for [`dcorrelator`](@ref) output (spin and electron alike), whose
  retarded `-im` prefactor flips the sign; the spectrum is then purely
  imaginary on the diagonal, `-imag(diag)/2`. See the inner
  [`fourier_kw`](@ref) method.

# Returns
Matrix of shape `(length(ws), length(ks))` whose `[w, k]` entry is the
group × group matrix `G(k, ω)/(4π²)` (a `1 × 1` matrix for the default
`regroup`); use `only(...)` or an extra trace/sum to extract a scalar spectrum.
"""
function fourier_kw(gf_rt::AbstractArray, rs::AbstractArray{<:AbstractArray}, ts::AbstractRange, ks::AbstractArray{<:AbstractArray}, ws::AbstractArray{<:Number};
                    mthreads::Integer=Threads.nthreads(), broadentype=(0.05, "G"), regroup::AbstractArray{<:AbstractArray}=[Vector(1:size(gf_rt,1)),], center::Union{Nothing,AbstractArray{<:Integer}}=nothing,
                    kernel::Symbol=:exp, symmetrize::Union{Bool,Integer}=false)
    @assert size(gf_rt, 1) == length(rs) "Dimension mismatch: the length of site positions 'rs' must equal to the size of green function matrix 'gf_rt'!"
    dampings = _damping_vector(ts, broadentype)
    gf_kw = Matrix(undef, length(ws), length(ks))
    for k in eachindex(ks)
        gf_kt = fourier_kt(gf_rt, rs, ks[k]; regroup=regroup, center=center, kernel=kernel)
        # Multi-threaded frequency loop using atomic counter
        idx = Threads.Atomic{Int}(1)
        Threads.@sync for _ in 1:mthreads
            Threads.@spawn while true
                w = Threads.atomic_add!(idx, 1)
                w > length(ws) && break
                gf_kw[w, k] = fourier_kw(gf_kt, ts, ws[w], dampings; symmetrize=symmetrize)
            end
        end
    end
    return gf_kw/(4π^2)
end

"""
    fourier_rw(gf_rt, ts, ws; broadentype=(0.05, "G"), mthreads=nthreads(), ifsum=true, symmetrize=false)

Time Fourier transform from `G(r, t)` to `G(r, ω)` (real-space, frequency domain).

# Arguments
- `gf_rt`: Green's function `(N_sites, N_sites, N_times)`.
- `ts`: time points.
- `ws`: frequency points.
- `broadentype`: broadening specification, e.g., `(0.05, "G")` or `(η, "GN")` for
  the window-normalized Gaussian `exp(-η·(t/tmax)^2)`.
- `mthreads`: number of threads.
- `ifsum`: if `true`, use simple summation; if `false`, use numerical integration.
- `symmetrize`: `false` (default), `true`/`+1`, or `-1`. If set, use the
  conjugation symmetry `G_{ba}(-t) = ±conj(G_{ab}(t))` (valid for
  `⟨A_a(t) A†_b(0)⟩` correlators in a time-translation invariant state) to
  fold the two-sided time integral onto the data window,

```math
G_{ab}(r, \\omega) = \\int_0^T dt\\, f(t) \\big[ e^{i\\omega t} G_{ab}(t) \\pm e^{-i\\omega t} G_{ba}(t)^* \\big] .
```

  With `+1` each frequency slice is manifestly Hermitian in `(a, b)` with a
  real diagonal; with `-1` it is manifestly anti-Hermitian with a purely
  imaginary diagonal. Either way, the unphysical truncation-induced component
  in the complementary quadrature is removed. Use `-1` for
  [`dcorrelator`](@ref) output (spin and electron alike), whose retarded
  `-im` prefactor flips the conjugation sign; the spectral function is then
  `-imag(diag)/2`. `ts` must start at `t = 0`, and the result is the
  two-sided transform (twice the corresponding quadrature of the one-sided
  one).

# Returns
Array of shape `(N_sites, N_sites, length(ws))`.
"""
function fourier_rw(gf_rt::AbstractArray, ts::AbstractArray, ws::AbstractArray; broadentype=(0.05, "G"), mthreads::Integer=Threads.nthreads(), ifsum::Bool=true, symmetrize::Union{Bool,Integer}=false)
    _check_gf_time_axes(gf_rt, ts)
    s = _sym_sign(symmetrize)
    (s != 0 && !iszero(first(ts))) && throw(ArgumentError("`symmetrize` requires time data starting at t = 0, got ts[1] = $(first(ts))."))
    (s != 0 && size(gf_rt, 1) != size(gf_rt, 2)) && throw(ArgumentError("`symmetrize` needs the transposed element G_{ba}(t), so gf_rt must be square (all source sites evolved). For a subset of source sites, use `fourier_kw` with `center`, where each element is folded with its own conjugate."))
    dampings = _damping_vector(ts, broadentype)
    gf_rw = zeros(ComplexF64, size(gf_rt, 1), size(gf_rt, 2), length(ws))
    if mthreads == 1
        for i in eachindex(ws)
            for a in axes(gf_rt, 1)
                for b in axes(gf_rt, 2)
                    temp = _time_integrand(gf_rt, a, b, ws[i], ts, dampings, s, true)
                    gf_rw[a,b,i] = ifsum ? sum(temp)*((ts[end]-ts[1])/length(ts)) : integrate(ts, temp)
                end
            end
        end
    else
        idx = Threads.Atomic{Int}(1)
        n = length(ws)
        Threads.@sync for _ in 1:mthreads
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1)
                i > n && break
                for a in axes(gf_rt, 1)
                    for b in axes(gf_rt, 2)
                        temp = _time_integrand(gf_rt, a, b, ws[i], ts, dampings, s, true)
                        gf_rw[a,b,i] = ifsum ? sum(temp)*((ts[end]-ts[1])/length(ts)) : integrate(ts, temp)
                    end
                end
            end
        end
    end
    return gf_rw
end

function _check_gf_time_axes(gf_rt::AbstractArray, ts::AbstractArray)
    ndims(gf_rt) >= 3 || throw(DimensionMismatch("gf_rt must have at least three axes, with time on the third axis."))
    size(gf_rt, 3) == length(ts) || throw(DimensionMismatch("The third axis of gf_rt must have the same length as ts."))
    length(ts) > 0 || throw(ArgumentError("ts must not be empty."))
    return nothing
end

function _damping_vector(ts::AbstractArray, broadentype)
    return isnothing(broadentype) ? ones(Float64, length(ts)) : [damping(t, broadentype; tmax=last(ts)-first(ts)) for t in ts]
end

"""
    _sym_sign(symmetrize)

Normalize the `symmetrize` keyword to a sign `s ∈ (-1, 0, +1)`: `false`/`0`
disables symmetrization, `true`/`+1` assumes `G(-t) = +conj(G(t))` (bare
correlators), `-1` assumes `G(-t) = -conj(G(t))` (Green's functions carrying
the retarded `-im` prefactor, e.g. [`dcorrelator`](@ref) output).
"""
function _sym_sign(symmetrize::Union{Bool,Integer})
    (symmetrize === false || symmetrize == 0) && return 0
    (symmetrize === true || symmetrize == 1) && return 1
    symmetrize == -1 && return -1
    throw(ArgumentError("Invalid symmetrize: $symmetrize (expected false, true, or ±1)"))
end

"""
    _time_integrand(gf, a, b, w, ts, dampings, s, transpose)

Integrand of the time Fourier transform for the `(a, b)` matrix element:
`f(t) e^{i w t} G_{ab}(t)` for `s == 0`, or the folded two-sided form
`f(t) [e^{i w t} G_{ab}(t) + s·e^{-i w t} conj(G_{pq}(t))]` for `s = ±1`,
which assumes `G_{pq}(-t) = s·conj(G_{ab}(t))` and time data starting at
`t = 0`. With `transpose=true` the conjugated partner is the transposed
element `(p, q) = (b, a)` (exact for a full square correlator); with
`transpose=false` it is the element itself (valid per element under
translation invariance, e.g. fixed-center displacement data from
[`fourier_kt`](@ref) with `center`).
"""
function _time_integrand(gf::AbstractArray, a, b, w::Number, ts::AbstractArray, dampings::AbstractArray, s::Integer, transpose::Bool)
    if s == 0
        return gf[a, b, :] .* cis.(w*ts) .* dampings
    end
    p, q = transpose ? (b, a) : (a, b)
    if s == 1
        return (gf[a, b, :] .* cis.(w*ts) .+ conj.(gf[p, q, :]) .* cis.(-w*ts)) .* dampings
    else
        return (gf[a, b, :] .* cis.(w*ts) .- conj.(gf[p, q, :]) .* cis.(-w*ts)) .* dampings
    end
end

function _time_integral(ts::AbstractArray, values::AbstractArray, ifsum::Bool)
    return ifsum ? sum(values)*((ts[end]-ts[1])/length(ts)) : integrate(ts, values)
end

"""
    fourier_rz(gf_rt, ts, zs; broadentype=nothing, mthreads=nthreads(), ifsum=false)

Time Fourier/Laplace transform from retarded real-time Green function `G(t)` to
complex frequency `G(z)`,

```math
G(z) = \\int_0^T dt\\, e^{i z t} G(t).
```

For imaginary-axis CPT grand-potential calculations, pass
`zs = μ .+ im .* iωs`, so the kernel becomes `exp(i*μ*t - iω*t)`.
`gf_rt` may be either the normal matrix or an already assembled Gorkov matrix.
"""
function fourier_rz(gf_rt::AbstractArray, ts::AbstractArray, zs::AbstractArray; broadentype=nothing, mthreads::Integer=Threads.nthreads(), ifsum::Bool=false)
    _check_gf_time_axes(gf_rt, ts)
    dampings = _damping_vector(ts, broadentype)
    gf_rz = zeros(ComplexF64, size(gf_rt, 1), size(gf_rt, 2), length(zs))
    if mthreads == 1
        for i in eachindex(zs)
            kernel = exp.(im*zs[i].*ts) .* dampings
            for a in axes(gf_rt, 1), b in axes(gf_rt, 2)
                gf_rz[a,b,i] = _time_integral(ts, gf_rt[a,b,:] .* kernel, ifsum)
            end
        end
    else
        idx = Threads.Atomic{Int}(1)
        n = length(zs)
        Threads.@sync for _ in 1:mthreads
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1)
                i > n && break
                kernel = exp.(im*zs[i].*ts) .* dampings
                for a in axes(gf_rt, 1), b in axes(gf_rt, 2)
                    gf_rz[a,b,i] = _time_integral(ts, gf_rt[a,b,:] .* kernel, ifsum)
                end
            end
        end
    end
    return gf_rz
end

"""
    fourier_riw(gf_rt, ts, iws; mu=0, kwargs...)

Convenience wrapper for `fourier_rz(gf_rt, ts, mu .+ im .* iws; kwargs...)`.
"""
function fourier_riw(gf_rt::AbstractArray, ts::AbstractArray, iws::AbstractArray; mu::Real=0, kwargs...)
    return fourier_rz(gf_rt, ts, mu .+ im .* iws; kwargs...)
end

"""
    static_structure_factor(ss, rs, ks)

Compute the static structure factor `S(k)` from real-space correlations `ss`.

``S(\\mathbf{k}) = \\frac{1}{N} \\sum_{a,b} \\langle S_a S_b \\rangle e^{i\\mathbf{k}\\cdot(\\mathbf{r}_a - \\mathbf{r}_b)}``

# Arguments
- `ss`: correlation matrix `(N_sites, N_sites)`.
- `rs`: site position vectors.
- `ks`: array of momentum vectors.
"""
function static_structure_factor(ss, rs, ks)
    sf = zeros(ComplexF64, length(ks))
    for i in eachindex(ks)
        for a in eachindex(rs), b in eachindex(rs)
            sf[i] += ss[a, b]*cis(dot(ks[i], rs[a]-rs[b]))
        end
    end
    return sf/length(rs)
end
