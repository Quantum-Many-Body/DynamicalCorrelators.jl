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
- `type_string` is one of `"G"` (Gaussian), `"L"` (Lorentzian), `"B"` (Blackman), `"P"` (Parzen).
- `parameter` is `η` for Gaussian/Lorentzian or `T` for window functions.
"""
function damping(t, broadentype)
    if broadentype[2] == "G"
        broaden = broaden_gauss
    elseif broadentype[2] == "L"
        broaden = broaden_lorentz
    elseif broadentype[2] == "B"
        broaden = blackman_window
    elseif broadentype[2] == "P"
        broaden = parzen_window
    else
        throw(ArgumentError("Invalid broadening type: $broadentype"))
    end
    return broaden(t, broadentype[1])
end

"""
    fourier_kt(gf_rt, rs, k; regroup=[collect(1:size(gf_rt,1))], center)

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
function fourier_kt(gf_rt::AbstractArray, rs::AbstractArray{<:AbstractArray}, k::AbstractArray{<:Number}; regroup::AbstractArray{<:AbstractArray}=[Vector(1:size(gf_rt,1)),], center)
    if isnothing(center)
        dest = zeros(ComplexF64, length(regroup), length(regroup), size(gf_rt, 3))
        for x in eachindex(regroup), y in eachindex(regroup), l in axes(gf_rt, 3)
            for j in eachindex(regroup[x]), i in eachindex(regroup[y])
                dest[x, y, l] += gf_rt[regroup[x][j], regroup[y][i], l]*cis(-dot(k, rs[regroup[y][i]]-rs[regroup[x][j]]))
            end
        end
        return dest
    else
        @assert size(gf_rt, 2) == length(center) "Invalid length of center and size(gf_rt, 2)!"
        dest = zeros(ComplexF64, length(regroup), length(center), size(gf_rt, 3))
        for x in eachindex(regroup), y in eachindex(center), l in axes(gf_rt, 3)
            for j in eachindex(regroup[x])
                dest[x, y, l] += gf_rt[regroup[x][j], y, l]*cis(-dot(k, rs[center[y]]-rs[regroup[x][j]]))
            end
        end
        return dest
    end
end

"""
    fourier_kw(gf_kt, ts, w, dampings)

Time Fourier transform of `G(k, t)` to `G(k, ω)` at a single frequency `w`,
with pre-computed damping factors.
"""
function fourier_kw(gf_kt::AbstractArray, ts::AbstractRange, w::Number, dampings::AbstractArray)
    dest = zeros(ComplexF64, size(gf_kt, 1), size(gf_kt, 2))
    for x in axes(gf_kt, 1), y in axes(gf_kt, 2)
        temp = gf_kt[x, y, :] .* cis.(w*ts) .* dampings
        dest[x, y] = integrate(ts, temp)
    end
    return dest
end

"""
    fourier_kw(gf_rt, rs, ts, ks, ws; mthreads=nthreads(), broadentype=(0.05, "G"), regroup=...)

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
- `broadentype`: broadening specification, e.g., `(0.05, "G")` for Gaussian with η=0.05.
- `regroup`: groups of MPS site indices defining the physical channels of the
  transform, forwarded to [`fourier_kt`](@ref) — see its docstring for the
  grouping rules and recipes (single group for 1:1 site mappings, one group per
  orbital/leg across all unit cells for band resolution, one group per unit
  for coarse graining).
- `center`: `nothing` (default) or the physical source-site indices when
  `gf_rt` was computed only for a subset of sources; forwarded to
  [`fourier_kt`](@ref).

# Returns
Matrix of shape `(length(ws), length(ks))` whose `[w, k]` entry is the
group × group matrix `G(k, ω)/(4π²)` (a `1 × 1` matrix for the default
`regroup`); use `only(...)` or an extra trace/sum to extract a scalar spectrum.
"""
function fourier_kw(gf_rt::AbstractArray, rs::AbstractArray{<:AbstractArray}, ts::AbstractRange, ks::AbstractArray{<:AbstractArray}, ws::AbstractArray{<:Number};
                    mthreads::Integer=Threads.nthreads(), broadentype=(0.05, "G"), regroup::AbstractArray{<:AbstractArray}=[Vector(1:size(gf_rt,1)),], center::Union{Nothing,AbstractArray{<:Integer}}=nothing)
    @assert size(gf_rt, 1) == length(rs) "Dimension mismatch: the length of site positions 'rs' must equal to the size of green function matrix 'gf_rt'!"
    dampings = [damping(t, broadentype) for t in ts]
    gf_kw = Matrix(undef, length(ws), length(ks))
    for k in eachindex(ks)
        gf_kt = fourier_kt(gf_rt, rs, ks[k]; regroup=regroup, center=center)
        # Multi-threaded frequency loop using atomic counter
        idx = Threads.Atomic{Int}(1)
        Threads.@sync for _ in 1:mthreads
            Threads.@spawn while true
                w = Threads.atomic_add!(idx, 1)
                w > length(ws) && break
                gf_kw[w, k] = fourier_kw(gf_kt, ts, ws[w], dampings)
            end
        end
    end
    return gf_kw/(4π^2)
end

"""
    fourier_rw(gf_rt, ts, ws; broadentype=(0.05, "G"), mthreads=nthreads(), ifsum=true)

Time Fourier transform from `G(r, t)` to `G(r, ω)` (real-space, frequency domain).

# Arguments
- `gf_rt`: Green's function `(N_sites, N_sites, N_times)`.
- `ts`: time points.
- `ws`: frequency points.
- `broadentype`: broadening specification.
- `mthreads`: number of threads.
- `ifsum`: if `true`, use simple summation; if `false`, use numerical integration.

# Returns
Array of shape `(N_sites, N_sites, length(ws))`.
"""
function fourier_rw(gf_rt::AbstractArray, ts::AbstractArray, ws::AbstractArray; broadentype=(0.05, "G"), mthreads::Integer=Threads.nthreads(), ifsum::Bool=true)
    _check_gf_time_axes(gf_rt, ts)
    dampings = [damping(t, broadentype) for t in ts]
    gf_rw = zeros(ComplexF64, size(gf_rt, 1), size(gf_rt, 2), length(ws))
    if mthreads == 1
        for i in eachindex(ws)
            for a in axes(gf_rt, 1)
                for b in axes(gf_rt, 2)
                    temp = gf_rt[a,b,:] .* cis.(ws[i]*ts).* dampings
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
                        temp = gf_rt[a,b,:] .* cis.(ws[i]*ts).* dampings
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
    return isnothing(broadentype) ? ones(Float64, length(ts)) : [damping(t, broadentype) for t in ts]
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
