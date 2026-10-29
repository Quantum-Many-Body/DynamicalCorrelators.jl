using Test
using SafeTestsets

@safetestset "DynamicalCorrelators" begin
    include("DynamicalCorrelators.jl")
end

@safetestset "MomentumSpace" begin
    include("MomentumSpace.jl")
end