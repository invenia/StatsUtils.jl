using StatsUtils
using Compat
using Compat.Test
using Compat.LinearAlgebra

import StatsUtils: sqrtcov, sqrtcor

@testset "StatsUtils" begin
    data = [
        0.6948  0.4387  0.1869;
        0.3171  0.3816  0.4898;
        0.9502  0.7655  0.4456;
        0.0344  0.7952  0.6463
    ]
    weights = [1, 1, 1, 1]

    @testset "std" begin
        @test StatsUtils.std(data, weights) ≈ vec(Compat.std(data, dims=1))
    end

    @testset "cov" begin
        @test StatsUtils.cov(data, weights) ≈ cov(data)
    end

    @testset "cor" begin
        @test StatsUtils.cor(data, weights) ≈ cor(data)
    end

    @testset "sqrtcov" begin
        X = sqrtcov(data, weights)
        @test X'X ≈ cov(data)
        @test sqrtcov(data) ≈ X
    end

    @testset "sqrtcor" begin
        expected = [
             0.2793 -0.4196 -0.7727;
            -0.2598 -0.5729  0.1442;
             0.6438  0.4565  0.0105;
            -0.6633  0.5360  0.6181
        ]

        X = sqrtcor(data, weights)
        @test isapprox(X, expected; rtol=1e-3)
        @test X'X ≈ cor(data)
        @test sqrtcor(data) ≈ X
    end

    @testset "sqrtcor to sqrtcov" begin
        sqrtcor_ = sqrtcor(data, weights)
        std_ = diagm(0 => StatsUtils.std(data, weights))
        sqrtcov_ = sqrtcov(sqrtcor_, std_)
        @test sqrtcov_' * sqrtcov_ ≈ cov(data)
    end
end
