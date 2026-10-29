using Test
using LinearAlgebra
using DynamicalCorrelators
using CairoMakie

function polygon_area(vertices)
    return abs(sum(vertices[i][1] * vertices[mod1(i + 1, length(vertices))][2] -
                   vertices[i][2] * vertices[mod1(i + 1, length(vertices))][1]
                   for i in eachindex(vertices))) / 2
end

@testset "Momentum-space geometry" begin
    cell = BravaisCell2D([1.0, 0.0], [0.5, sqrt(3) / 2])
    K = [2 / 3, 1 / 3]

    @test cell.B' * cell.A ≈ 2π * I
    bz = brillouin_zone(cell)
    @test length(bz) == 6
    @test polygon_area(bz) ≈ abs(det(cell.B))

    yc3 = MomentumGeometry(cell, TwistedTranslation((0, 3)))
    yc4 = MomentumGeometry(cell, TwistedTranslation((0, 4)))
    yc6 = MomentumGeometry(cell, TwistedTranslation((0, 6)))
    xc4 = MomentumGeometry(cell, TwistedTranslation((2, 2)))

    @test length(allowed_cuts(yc3)) == 3
    @test length(allowed_cuts(yc6)) == 6
    @test length(allowed_cuts(xc4)) == 2
    @test is_allowed(yc3, K)
    @test is_allowed(yc6, K)
    @test !is_allowed(yc4, K)
    @test is_allowed(xc4, K)
    @test nearest_cut(yc6, K).distance ≈ 0.0 atol=1e-12
    @test nearest_cut(yc4, K).distance > 0.1

    apbc = MomentumGeometry(cell, TwistedTranslation((0, 6); phase=π))
    periodic_offsets = sort(getproperty.(allowed_cuts(yc6), :offset))
    apbc_offsets = sort(getproperty.(allowed_cuts(apbc), :offset))
    @test apbc_offsets ≈ periodic_offsets .+ 1 / 12

    skew = MomentumGeometry(cell, TwistedTranslation((1, 3)))
    @test length(allowed_cuts(skew)) == 1
    @test length(bz_segments(skew)) > 1
    @test all(segment -> segment.sector == 0, bz_segments(skew))

    θ = 0.73
    up = MomentumGeometry(cell, TwistedTranslation((0, 6); phase=θ / 2))
    down = MomentumGeometry(cell, TwistedTranslation((0, 6); phase=-θ / 2))
    @test allowed_cuts(up)[1].offset ≈ θ / (24π)
    @test allowed_cuts(down)[1].offset ≈ -θ / (24π)
    @test all(cut -> is_allowed(up, cut.fractional; space=:magnetic), allowed_cuts(up))
    @test all(cut -> is_allowed(down, cut.fractional; space=:magnetic), allowed_cuts(down))

    torus = MomentumGeometry(cell, TwistedTranslation((0, 6)), TwistedTranslation((36, 0)))
    points = allowed_points(torus)
    @test length(points) == 216
    @test length(unique([round.(point.fractional; digits=10) for point in points])) == 216
    @test count(point -> isapprox(point.fractional, K; atol=1e-10), points) == 1
    @test all(point -> is_allowed(torus, point.fractional; space=:magnetic), points)

    for V in ([2 1; 1 3], [3 1; -1 2], [2 2; 1 3]), phase in (0.0, 0.37)
        geometry = MomentumGeometry(
            cell,
            TwistedTranslation((V[1, 1], V[1, 2]); phase),
            TwistedTranslation((V[2, 1], V[2, 2]); phase=-phase),
        )
        sampled = allowed_points(geometry)
        @test length(sampled) == abs(det(V))
        @test all(point -> is_allowed(geometry, point.fractional; space=:magnetic), sampled)
    end

    magnetic = Supercell2D(cell, [1 0; 0 2])
    magnetic_cylinder = MomentumGeometry(magnetic, TwistedTranslation((0, 6)))
    @test length(allowed_cuts(magnetic_cylinder)) == 3
    @test polygon_area(brillouin_zone(magnetic)) ≈ abs(det(cell.B)) / 2
    @test_throws ArgumentError allowed_cuts(MomentumGeometry(magnetic, TwistedTranslation((0, 5))))

    magnetic_torus = MomentumGeometry(magnetic, TwistedTranslation((0, 6)), TwistedTranslation((36, 0)))
    @test length(allowed_points(magnetic_torus)) == 108

    @test_throws ArgumentError TwistedTranslation((0, 0))
    @test_throws ArgumentError allowed_points(yc6)
    @test_throws ArgumentError allowed_points(MomentumGeometry(cell, TwistedTranslation((1, 1)), TwistedTranslation((2, 2))))
end

@testset "Momentum-space plotting" begin
    cell = BravaisCell2D([1.0, 0.0], [0.5, sqrt(3) / 2])
    cylinder = MomentumGeometry(cell, TwistedTranslation((0, 6)))
    torus = MomentumGeometry(cell, TwistedTranslation((0, 6)), TwistedTranslation((36, 0)))
    magnetic = MomentumGeometry(Supercell2D(cell, [1 0; 0 2]), TwistedTranslation((0, 6)))

    mktempdir() do directory
        cylinder_figure = momentum_plot(cylinder; markers=Dict(:K => [2 / 3, 1 / 3]))
        CairoMakie.save(joinpath(directory, "yc6.png"), cylinder_figure)
        @test filesize(joinpath(directory, "yc6.png")) > 0

        torus_figure = momentum_plot(torus)
        CairoMakie.save(joinpath(directory, "torus.png"), torus_figure)
        @test filesize(joinpath(directory, "torus.png")) > 0

        magnetic_figure = momentum_plot(magnetic; overlay_parent=true)
        @test magnetic_figure isa CairoMakie.Figure
        @test momentum_plot(magnetic; space=:parent) isa CairoMakie.Figure
        @test momentum_plot(cylinder; tiled=true) isa CairoMakie.Figure
    end
end
