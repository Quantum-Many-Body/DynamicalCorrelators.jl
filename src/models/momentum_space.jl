"""
    BravaisCell2D(a1, a2)
    BravaisCell2D(A)

Two-dimensional Bravais cell. `A` stores the real-space primitive vectors as
columns and the reciprocal vectors as columns of `B = 2π * inv(A)'`.
"""
struct BravaisCell2D
    A::Matrix{Float64}
    B::Matrix{Float64}
end

function BravaisCell2D(A::AbstractMatrix{<:Real})
    size(A) == (2, 2) || throw(DimensionMismatch("a two-dimensional Bravais cell needs a 2×2 matrix"))
    Af = Matrix{Float64}(A)
    abs(det(Af)) > 1e-14 || throw(ArgumentError("the Bravais vectors must be linearly independent"))
    return BravaisCell2D(Af, 2π * inv(Af'))
end

BravaisCell2D(a1::AbstractVector{<:Real}, a2::AbstractVector{<:Real}) = BravaisCell2D(hcat(a1, a2))
BravaisCell2D(vectors::AbstractVector{<:AbstractVector{<:Real}}) = length(vectors) == 2 ? BravaisCell2D(vectors[1], vectors[2]) : throw(ArgumentError("expected exactly two Bravais vectors"))

"""
    Supercell2D(parent, transform)

Describe an enlarged (for example magnetic) cell by `A_cell = A_parent * transform`.
The columns of the nonsingular integer matrix `transform` are the cell vectors in
parent-cell coordinates. Cylinder wrapping vectors remain expressed in the parent
basis when a `Supercell2D` is used in `MomentumGeometry`.
"""
struct Supercell2D
    parent::BravaisCell2D
    transform::Matrix{Int}
    cell::BravaisCell2D
end

function Supercell2D(parent::BravaisCell2D, transform::AbstractMatrix{<:Integer})
    size(transform) == (2, 2) || throw(DimensionMismatch("the supercell transform must be 2×2"))
    S = Matrix{Int}(transform)
    det(S) != 0 || throw(ArgumentError("the supercell transform must be nonsingular"))
    return Supercell2D(parent, S, BravaisCell2D(parent.A * S))
end

"""
    TwistedTranslation(coeffs; phase=0.0)

A compact direction with wrapping vector `W = coeffs[1] * a1 + coeffs[2] * a2`.
The boundary convention is `ψ(r + W) = exp(im * phase) * ψ(r)`.
"""
struct TwistedTranslation
    coeffs::NTuple{2,Int}
    phase::Float64
end

function TwistedTranslation(coeffs::NTuple{2,<:Integer}; phase::Real=0.0)
    L = (Int(coeffs[1]), Int(coeffs[2]))
    L != (0, 0) || throw(ArgumentError("a compact direction needs a nonzero wrapping vector"))
    isfinite(phase) || throw(ArgumentError("the boundary phase must be finite"))
    return TwistedTranslation(L, Float64(phase))
end

TwistedTranslation(coeffs::AbstractVector{<:Integer}; phase::Real=0.0) = length(coeffs) == 2 ? TwistedTranslation((coeffs[1], coeffs[2]); phase) : throw(ArgumentError("expected two wrapping coefficients"))

"""
    MomentumGeometry(cell, boundary...)
    MomentumGeometry(supercell, boundary...)

Momentum geometry with one compact direction (a cylinder) or two independent
compact directions (a periodic cluster). Wrapping coefficients always refer to
the parent Bravais cell.
"""
struct MomentumGeometry
    cell::Union{BravaisCell2D,Supercell2D}
    boundaries::Vector{TwistedTranslation}
end

function MomentumGeometry(cell::Union{BravaisCell2D,Supercell2D}, boundaries::TwistedTranslation...)
    length(boundaries) in (1, 2) || throw(ArgumentError("expected one or two compact directions"))
    return MomentumGeometry(cell, collect(boundaries))
end

_parent_cell(cell::BravaisCell2D) = cell
_parent_cell(cell::Supercell2D) = cell.parent
_working_cell(cell::BravaisCell2D) = cell
_working_cell(cell::Supercell2D) = cell.cell
_parent_cell(geometry::MomentumGeometry) = _parent_cell(geometry.cell)
_working_cell(geometry::MomentumGeometry) = _working_cell(geometry.cell)

function _working_coeffs(cell::BravaisCell2D, boundary::TwistedTranslation)
    return boundary.coeffs
end

function _working_coeffs(cell::Supercell2D, boundary::TwistedTranslation)
    L = cell.transform \ collect(Float64, boundary.coeffs)
    Lr = round.(L)
    all(isapprox.(L, Lr; atol=1e-10, rtol=0)) || throw(ArgumentError("wrapping vector $(boundary.coeffs) is not a translation of the enlarged cell"))
    return (Int(Lr[1]), Int(Lr[2]))
end

_working_coeffs(geometry::MomentumGeometry, boundary::TwistedTranslation) = _working_coeffs(geometry.cell, boundary)

struct MomentumCut
    sector::Int
    offset::Float64
    normal::NTuple{2,Int}
    fractional::Vector{Float64}
    parent_fractional::Vector{Float64}
    point::Vector{Float64}
    direction::Vector{Float64}
end

struct MomentumSegment
    sector::Int
    first::Vector{Float64}
    last::Vector{Float64}
end

struct MomentumPoint
    fractional::Vector{Float64}
    parent_fractional::Vector{Float64}
    cartesian::Vector{Float64}
    sectors::NTuple{2,Int}
end

function _to_parent_fractional(cell::BravaisCell2D, κ::AbstractVector)
    return Vector{Float64}(κ)
end

function _to_parent_fractional(cell::Supercell2D, κ::AbstractVector)
    return cell.transform' \ Vector{Float64}(κ)
end

function _to_working_fractional(cell::BravaisCell2D, κ::AbstractVector, space::Symbol)
    space in (:parent, :magnetic, :cell) || throw(ArgumentError("space must be :parent, :magnetic, :cell, or :cartesian"))
    return Vector{Float64}(κ)
end

function _to_working_fractional(cell::Supercell2D, κ::AbstractVector, space::Symbol)
    space in (:parent, :magnetic, :cell) || throw(ArgumentError("space must be :parent, :magnetic, :cell, or :cartesian"))
    return space === :parent ? cell.transform' * Vector{Float64}(κ) : Vector{Float64}(κ)
end

function _working_fractional(geometry::MomentumGeometry, momentum::AbstractVector{<:Real}, space::Symbol)
    length(momentum) == 2 || throw(DimensionMismatch("a momentum needs two components"))
    if space === :cartesian
        return _working_cell(geometry).B \ Vector{Float64}(momentum)
    end
    return _to_working_fractional(geometry.cell, momentum, space)
end

"""
    allowed_cuts(geometry)

Return the inequivalent momentum lines for a one-boundary cylinder. Every line
carries a sector label; folded segments from the same line retain this label.
"""
function allowed_cuts(geometry::MomentumGeometry)
    length(geometry.boundaries) == 1 || throw(ArgumentError("allowed_cuts requires exactly one compact direction"))
    boundary = only(geometry.boundaries)
    L = _working_coeffs(geometry, boundary)
    g = gcd(L...)
    n = (L[1] ÷ g, L[2] ÷ g)
    nrm2 = n[1]^2 + n[2]^2
    B = _working_cell(geometry).B
    flux = boundary.phase / (2π)
    cuts = MomentumCut[]
    for sector in 0:(g - 1)
        offset = (sector + flux) / g
        κ = offset .* collect(n) ./ nrm2
        tangent = [-n[2], n[1]]
        direction = B * tangent
        direction ./= norm(direction)
        push!(cuts, MomentumCut(sector, offset, n, κ, _to_parent_fractional(geometry.cell, κ), B * κ, direction))
    end
    return cuts
end

function _cross2(u, v)
    return u[1] * v[2] - u[2] * v[1]
end

function _polygon_area(vertices)
    return abs(sum(_cross2(vertices[i], vertices[mod1(i + 1, length(vertices))]) for i in eachindex(vertices))) / 2
end

function _unique_points!(points, point; tol=1e-9)
    any(p -> norm(p - point) <= tol, points) || push!(points, point)
    return points
end

function _bz_vertices(B, shell; tol=1e-10)
    constraints = Tuple{Vector{Float64},Float64}[]
    for i in (-shell):shell, j in (-shell):shell
        (i == 0 && j == 0) && continue
        G = B * [i, j]
        push!(constraints, (G, dot(G, G) / 2))
    end
    vertices = Vector{Float64}[]
    for i in eachindex(constraints), j in (i + 1):length(constraints)
        Gi, bi = constraints[i]
        Gj, bj = constraints[j]
        M = [Gi[1] Gi[2]; Gj[1] Gj[2]]
        abs(det(M)) <= tol && continue
        k = M \ [bi, bj]
        all(c -> dot(c[1], k) <= c[2] + tol, constraints) || continue
        _unique_points!(vertices, k; tol)
    end
    sort!(vertices; by=v -> atan(v[2], v[1]))
    return vertices
end

function _same_point_set(a, b; tol=1e-8)
    length(a) == length(b) || return false
    return all(p -> any(q -> norm(p - q) <= tol, b), a)
end

"""
    brillouin_zone(cell; max_shell=8, tol=1e-10)

Return counterclockwise Cartesian vertices of the first Brillouin zone, computed
as the Wigner–Seitz cell of the reciprocal lattice.
"""
function brillouin_zone(cell::Union{BravaisCell2D,Supercell2D}; max_shell::Integer=8, tol::Real=1e-10)
    B = _working_cell(cell).B
    previous = Vector{Float64}[]
    for shell in 1:max_shell
        vertices = _bz_vertices(B, shell; tol)
        if shell > 1 && _same_point_set(vertices, previous; tol=10tol)
            length(vertices) >= 3 || throw(ErrorException("failed to construct the Brillouin zone"))
            expected = abs(det(B))
            isapprox(_polygon_area(vertices), expected; atol=1e-8 * max(1, expected), rtol=1e-8) || throw(ErrorException("Brillouin-zone area check failed"))
            return vertices
        end
        previous = vertices
    end
    throw(ErrorException("Brillouin-zone construction did not converge; increase max_shell"))
end

brillouin_zone(geometry::MomentumGeometry; kwargs...) = brillouin_zone(geometry.cell; kwargs...)

function _inside_polygon(point, vertices; tol=1e-9)
    return all(i -> _cross2(vertices[mod1(i + 1, length(vertices))] - vertices[i], point - vertices[i]) >= -tol, eachindex(vertices))
end

function _clip_line_polygon(cut::MomentumCut, vertices; tol=1e-10)
    tmin, tmax = -Inf, Inf
    for i in eachindex(vertices)
        v = vertices[i]
        edge = vertices[mod1(i + 1, length(vertices))] - v
        c = _cross2(edge, cut.direction)
        d = _cross2(edge, cut.point - v)
        if abs(c) <= tol
            d >= -tol || return nothing
        elseif c > 0
            tmin = max(tmin, -d / c)
        else
            tmax = min(tmax, -d / c)
        end
        tmax >= tmin - tol || return nothing
    end
    isfinite(tmin) && isfinite(tmax) || return nothing
    tmax - tmin > tol || return nothing
    return (cut.point + tmin .* cut.direction, cut.point + tmax .* cut.direction)
end

function _same_segment(a::MomentumSegment, b::MomentumSegment; tol=1e-8)
    direct = norm(a.first - b.first) <= tol && norm(a.last - b.last) <= tol
    reverse = norm(a.first - b.last) <= tol && norm(a.last - b.first) <= tol
    return a.sector == b.sector && (direct || reverse)
end

function _push_segment!(segments, sector, first, last; tol=1e-8)
    segment = MomentumSegment(sector, first, last)
    any(s -> _same_segment(s, segment; tol), segments) || push!(segments, segment)
    return segments
end

function _same_segment_set(a, b; tol=1e-8)
    length(a) == length(b) || return false
    return all(s -> any(t -> _same_segment(s, t; tol), b), a)
end

function _default_space(::BravaisCell2D)
    return :parent
end

function _default_space(::Supercell2D)
    return :magnetic
end

function _space_cell(geometry::MomentumGeometry, space::Symbol)
    space in (:parent, :magnetic, :cell) || throw(ArgumentError("space must be :parent, :magnetic, or :cell"))
    return space === :parent ? _parent_cell(geometry) : _working_cell(geometry)
end

"""
    bz_segments(geometry; space=:magnetic, max_shell=6)

Clip the allowed lines against translated Brillouin zones and fold the segments
back into the selected zone. `space=:parent` selects the parent zone; `:magnetic`
and `:cell` select the working (possibly enlarged) zone.
"""
function bz_segments(geometry::MomentumGeometry; space::Symbol=_default_space(geometry.cell), max_shell::Integer=6, tol::Real=1e-9)
    cuts = allowed_cuts(geometry)
    cell = _space_cell(geometry, space)
    vertices = brillouin_zone(cell)
    previous = MomentumSegment[]
    for shell in 0:max_shell
        segments = MomentumSegment[]
        for i in (-shell):shell, j in (-shell):shell
            G = cell.B * [i, j]
            translated = [v + G for v in vertices]
            for cut in cuts
                clipped = _clip_line_polygon(cut, translated; tol)
                isnothing(clipped) && continue
                first, last = clipped
                _push_segment!(segments, cut.sector, first - G, last - G; tol=10tol)
            end
        end
        shell > 0 && _same_segment_set(segments, previous; tol=10tol) && return segments
        previous = segments
    end
    throw(ErrorException("line clipping did not converge; increase max_shell"))
end

function _fold01(x, tol)
    y = mod(x, 1.0)
    return (y <= tol || y >= 1 - tol) ? 0.0 : y
end

function _fold_to_bz(point, B, vertices; max_shell=8, tol=1e-9)
    for shell in 0:max_shell
        candidates = Vector{Float64}[]
        for i in (-shell):shell, j in (-shell):shell
            candidate = point - B * [i, j]
            _inside_polygon(candidate, vertices; tol) && push!(candidates, candidate)
        end
        isempty(candidates) || return candidates[argmin(norm.(candidates))]
    end
    throw(ErrorException("failed to fold momentum into the first Brillouin zone"))
end

"""
    allowed_points(geometry; tol=1e-10)

Enumerate unique momenta for a two-boundary periodic cluster. The number of
points is checked against the absolute determinant of the wrapping matrix.
"""
function allowed_points(geometry::MomentumGeometry; tol::Real=1e-10)
    length(geometry.boundaries) == 2 || throw(ArgumentError("allowed_points requires exactly two compact directions"))
    b1, b2 = geometry.boundaries
    L1 = _working_coeffs(geometry, b1)
    L2 = _working_coeffs(geometry, b2)
    V = [L1[1] L1[2]; L2[1] L2[2]]
    N = abs(det(V))
    N != 0 || throw(ArgumentError("the two compact directions must be linearly independent"))
    flux = [b1.phase, b2.phase] ./ (2π)
    corners = [V * [i, j] - flux for i in (0, 1), j in (0, 1)]
    lo = [floor(Int, minimum(c -> c[d], corners)) - 1 for d in 1:2]
    hi = [ceil(Int, maximum(c -> c[d], corners)) + 1 for d in 1:2]
    B = _working_cell(geometry).B
    vertices = brillouin_zone(geometry.cell)
    points = Dict{NTuple{2,Int},MomentumPoint}()
    for l in lo[1]:hi[1], m in lo[2]:hi[2]
        integers = [l, m]
        κraw = V \ (integers + flux)
        κ = _fold01.(κraw, tol)
        key = (round(Int, κ[1] / tol), round(Int, κ[2] / tol))
        haskey(points, key) && continue
        cartesian = _fold_to_bz(B * κ, B, vertices)
        sectors = (mod(l, gcd(L1...)), mod(m, gcd(L2...)))
        points[key] = MomentumPoint(κ, _to_parent_fractional(geometry.cell, κ), cartesian, sectors)
    end
    length(points) == N || throw(ErrorException("momentum enumeration produced $(length(points)) unique points instead of $N; adjust tol"))
    result = collect(values(points))
    sort!(result; by=point -> (point.sectors, point.fractional))
    return result
end

"""
    is_allowed(geometry, momentum; space=:parent, tol=1e-9)

Check whether a momentum lies on the allowed cylinder line set, or satisfies both
quantization conditions of a periodic cluster. Fractional input is interpreted
in the parent basis unless `space=:magnetic`, `:cell`, or `:cartesian` is given.
"""
function is_allowed(geometry::MomentumGeometry, momentum::AbstractVector{<:Real}; space::Symbol=:parent, tol::Real=1e-9)
    κ = _working_fractional(geometry, momentum, space)
    return all(geometry.boundaries) do boundary
        L = _working_coeffs(geometry, boundary)
        value = dot(collect(L), κ) - boundary.phase / (2π)
        isapprox(value, round(value); atol=tol, rtol=0)
    end
end

"""
    nearest_cut(geometry, momentum; space=:parent)

Return the closest allowed line and its perpendicular Cartesian distance.
"""
function nearest_cut(geometry::MomentumGeometry, momentum::AbstractVector{<:Real}; space::Symbol=:parent)
    κ = _working_fractional(geometry, momentum, space)
    k = _working_cell(geometry).B * κ
    cuts = allowed_cuts(geometry)
    distances = map(cuts) do cut
        δ = k - cut.point
        norm(δ - dot(δ, cut.direction) * cut.direction)
    end
    index = argmin(distances)
    return (cut=cuts[index], distance=distances[index])
end

"""
    momentum_plot(geometry; kwargs...)

Plot Brillouin-zone boundaries, sector-resolved momentum cuts or discrete points.
This method is provided by the CairoMakie package extension after
`using CairoMakie`.
"""
function momentum_plot end
