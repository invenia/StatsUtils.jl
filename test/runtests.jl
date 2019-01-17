using StatsUtils
using Compat
using Compat.Test
using Compat.LinearAlgebra
using Compat.Statistics

using StatsBase
using StatsUtils: sqrtcov, sqrtcor

@testset "StatsUtils" begin
    data = [
        0.6948  0.4387  0.1869;
        0.3171  0.3816  0.4898;
        0.9502  0.7655  0.4456;
        0.0344  0.7952  0.6463
    ]
    w1 = ones(4)
    w2 = [0.25, 0.33, 0.81, 0.14]

    @testset "std" begin
        @test StatsUtils.std(data, w1) ≈ vec(Compat.Statistics.std(data, dims=1))
        @test StatsUtils.std(data, w2; corrected=false) ≈ vec(StatsBase.std(data, StatsBase.weights(w2), 1; corrected=false))
    end

    @testset "cov" begin
        @test StatsUtils.cov(data, w1) ≈ Compat.Statistics.cov(data)
        @test StatsUtils.cov(data, w2; corrected=false) ≈ StatsBase.cov(data, StatsBase.weights(w2), 1; corrected=false)
    end

    @testset "cor" begin
        @test StatsUtils.cor(data, w1) ≈ Compat.Statistics.cor(data)
        @test StatsUtils.cor(data, w2) ≈ StatsBase.cor(data, StatsBase.weights(w2), 1)
    end

    @testset "sqrtcov" begin
        X = sqrtcov(data, w1)
        @test X'X ≈ Compat.Statistics.cov(data)
        @test sqrtcov(data) ≈ X

        X = sqrtcov(data, w2; corrected=false)
        @test X'X ≈ StatsBase.cov(data, StatsBase.weights(w2), 1; corrected=false)
    end

    @testset "sqrtcor" begin
        expected = [
             0.2793 -0.4196 -0.7727;
            -0.2598 -0.5729  0.1442;
             0.6438  0.4565  0.0105;
            -0.6633  0.5360  0.6181
        ]

        X = sqrtcor(data, w1)
        @test isapprox(X, expected; rtol=1e-3)
        @test X'X ≈ Compat.Statistics.cor(data)
        @test sqrtcor(data) ≈ X

        X = sqrtcor(data, w2)
        @test X'X ≈ StatsBase.cor(data, StatsBase.weights(w2))
    end

    @testset "sqrtcor to sqrtcov" begin
        sqrtcor_ = sqrtcor(data, w1)
        std_ = diagm(0 => StatsUtils.std(data, w1))
        sqrtcov_ = sqrtcov(sqrtcor_, std_)
        @test sqrtcov_' * sqrtcov_ ≈ Compat.Statistics.cov(data)
    end
end
