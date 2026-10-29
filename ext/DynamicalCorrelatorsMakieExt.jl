module DynamicalCorrelatorsMakieExt

__precompile__(false)

using DynamicalCorrelators
import CairoMakie

const DC = DynamicalCorrelators
const MK = CairoMakie

function _closed(vertices)
    return vcat(vertices, [first(vertices)])
end

function _draw_polygon!(ax, vertices, shift; color=:black, linewidth=1.6, linestyle=:solid)
    points = _closed([v + shift for v in vertices])
    MK.lines!(ax, first.(points), last.(points); color, linewidth, linestyle)
end

function _point_in_space(point, geometry, cell, vertices, space)
    if space !== :parent
        return point.cartesian
    end
    return DC._fold_to_bz(point.cartesian, cell.B, vertices)
end

function DC.momentum_plot(geometry::DC.MomentumGeometry;
                          space::Symbol=DC._default_space(geometry.cell),
                          tiled::Bool=false,
                          show_bz::Bool=true,
                          show_points::Bool=true,
                          overlay_parent::Bool=false,
                          markers=nothing,
                          marker_space::Symbol=:parent,
                          sector_colors=nothing,
                          size=(760, 680),
                          title::AbstractString="")
    cell = DC._space_cell(geometry, space)
    vertices = DC.brillouin_zone(cell)
    shifts = tiled ? vec([cell.B * [i, j] for i in -1:1, j in -1:1]) : [zeros(2)]
    palette = isnothing(sector_colors) ? [:blue, :red, :green, :orange, :purple, :brown, :pink, :olive, :cyan, :magenta, :gray, :black] : sector_colors
    fig = MK.Figure(; size)
    ax = MK.Axis(fig[1, 1]; aspect=MK.DataAspect(), xlabel="kₓ", ylabel="kᵧ", title,
                 xgridvisible=false, ygridvisible=false)

    if show_bz
        for shift in shifts
            _draw_polygon!(ax, vertices, shift)
        end
    end

    if length(geometry.boundaries) == 1
        segments = DC.bz_segments(geometry; space)
        cuts = DC.allowed_cuts(geometry)
        for cut in cuts
            color = palette[mod1(cut.sector + 1, length(palette))]
            for segment in filter(s -> s.sector == cut.sector, segments), shift in shifts
                first = segment.first + shift
                last = segment.last + shift
                MK.lines!(ax, [first[1], last[1]], [first[2], last[2]]; color, linewidth=2.0)
            end
        end
    elseif show_points
        points = DC.allowed_points(geometry)
        for shift in shifts
            cartesian = [_point_in_space(point, geometry, cell, vertices, space) + shift for point in points]
            MK.scatter!(ax, first.(cartesian), last.(cartesian); color=:navy, markersize=7)
        end
    end

    if overlay_parent && geometry.cell isa DC.Supercell2D && space !== :parent
        parent_vertices = DC.brillouin_zone(DC._parent_cell(geometry))
        _draw_polygon!(ax, parent_vertices, zeros(2); color=:gray30, linewidth=1.4, linestyle=:dash)
    end

    if !isnothing(markers)
        for (label, momentum) in pairs(markers)
            κ = DC._working_fractional(geometry, momentum, marker_space)
            k = DC._working_cell(geometry).B * κ
            k = DC._fold_to_bz(k, cell.B, vertices)
            for shift in shifts
                point = k + shift
                MK.scatter!(ax, [point[1]], [point[2]]; color=:red, marker=:star5, markersize=17)
                MK.text!(ax, point[1], point[2]; text=String(label), color=:red, align=(:left, :bottom), fontsize=16)
            end
        end
    end

    return fig
end

end
