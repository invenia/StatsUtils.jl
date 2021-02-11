using Distances
using Distributions
using Distributions: GenericMvTDist
using IndexedDistributions
using LinearAlgebra
using PDMats
using PDMatsExtras
using Random
using Statistics
using StatsBase
using StatsUtils
using StatsUtils: sqrtcov, sqrtcor
using Test

@testset "StatsUtils" begin
    data = [
        0.6948  0.4387  0.1869;
        0.3171  0.3816  0.4898;
        0.9502  0.7655  0.4456;
        0.0344  0.7952  0.6463
    ]
    w1 = ones(4)
    w2 = [0.25, 0.33, 0.81, 0.14]

    # for testsets below
    B = (reshape(2:10, 3, 3) / 12) .^ 2
    A = B' * B

    # Woodbury Matrix
    D = Diagonal([1.,2,3])
    S = Diagonal([4.,5,6])
    W = WoodburyPDMat(B, D, S)

    @testset "std" begin
        @test StatsUtils.std(data, w1) ≈ vec(Statistics.std(data, dims=1))
        @test StatsUtils.std(data, w2; corrected=false) ≈ vec(StatsBase.std(data, StatsBase.weights(w2), 1; corrected=false))
    end

    @testset "cov" begin
        # on data
        @test StatsUtils.cov(data, w1) ≈ Statistics.cov(data)
        @test StatsUtils.cov(data, w2; corrected=false) ≈ StatsBase.cov(data, StatsBase.weights(w2), 1; corrected=false)

        # on distribution
        @testset "AbstractPDMat type $(typeof(pd))" for pd in [
            PDiagMat(diag(D)), PDMat(Symmetric(A)), PSDMat(Symmetric(A)), W
        ]
            @testset "distribution type $(typeof(dist))" for dist in [
                MvNormal(ones(size(pd, 1)), pd),
                GenericMvTDist(2.2, ones(size(pd, 1)), pd),
            ]
                # the resulting covariance should preserve the AbstractPDMat type
                @test typeof(StatsUtils.cov(dist)) == typeof(pd)

                @testset "IndexedDistribution" begin
                    names = "n" .* string.(collect(1:length(dist)))
                    id = IndexedDistribution(dist, names)
                    @test typeof(StatsUtils.cov(dist)) == typeof(pd)
                end

            end
            # throw error when the dof of MvT is < 2
            @test_throws ArgumentError StatsUtils.cov(GenericMvTDist(1.8, ones(size(pd, 1)), pd))
        end
    end

    @testset "scale" begin
        @testset "AbstractPDMat type $(typeof(pd))" for pd in [
            PDiagMat(diag(D)), PDMat(Symmetric(A)), PSDMat(Symmetric(A)), W
        ]
            @testset "distribution type $(typeof(dist))" for dist in [
                MvNormal(ones(size(pd, 1)), pd),
                GenericMvTDist(2.2, ones(size(pd, 1)), pd),
            ]
                # the resulting covariance should preserve the AbstractPDMat type
                @test typeof(StatsUtils.scale(dist)) == typeof(pd)

                @testset "IndexedDistribution" begin
                    names = "n" .* string.(collect(1:length(dist)))
                    id = IndexedDistribution(dist, names)
                    @test typeof(StatsUtils.scale(dist)) == typeof(pd)
                end
            end
        end
    end

    @testset "cor" begin
        @test StatsUtils.cor(data, w1) ≈ Statistics.cor(data)
        @test StatsUtils.cor(data, w2) ≈ StatsBase.cor(data, StatsBase.weights(w2), 1)
    end

    @testset "sqrtcov" begin
        X = sqrtcov(data, w1)
        @test X'X ≈ Statistics.cov(data)
        @test sqrtcov(data) ≈ X

        X = sqrtcov(data, w2; corrected=false)
        @test X'X ≈ StatsBase.cov(data, StatsBase.weights(w2), 1; corrected=false)

        @testset "PSDMat" begin
            chol_piv = cholesky(A, Val(true))
            Σ_psd = PSDMat(chol_piv)
            @test isequal(sqrtcov(chol_piv), sqrtcov(Σ_psd))
        end

        @testset "PDMat" begin
            chol = cholesky(A, Val(false))
            Σ_pd = PDMat(chol)
            @test isequal(sqrtcov(chol), sqrtcov(Σ_pd))
        end

        @testset "WoodburyPDMat" begin
            Σ_pd = PSDMat(Matrix(Symmetric(W)))
            @test isequal(sqrtcov(W), sqrtcov(Σ_pd))
            @test sqrtcov(W)' * sqrtcov(W) ≈ W
            chol = cholesky(W' * W, Val(false))
            Σ_pd = PDMat(chol)
            @test isequal(sqrtcov(chol), sqrtcov(Σ_pd))

        end

        @testset "Factorizations" begin
            chol_fact = cholesky(A, Val(false))
            @test sqrtcov(chol_fact) == chol_fact.U

            qr_fact = qr(A)
            @test sqrtcov(qr_fact) == qr_fact.R
        end

        @testset "MvNormal" begin
            # FullNormal
            dist = MvNormal(ones(size(A, 1)), A)
            chol = cholesky(A)
            @test isequal(sqrtcov(dist), sqrtcov(chol))

            # DiagNormal
            Σ = Diagonal(0.1:0.1:0.5)
            dist = MvNormal(ones(size(Σ, 1)), Σ)
            @test isequal(sqrtcov(dist), sqrt(Distributions.cov(dist)))
            @test isequal(sqrtcov(dist), sqrt(Σ))

            # WoodburyPDMat
            dist = MvNormal(ones(size(W, 1)), W)
            @test isequal(sqrtcov(dist), sqrtcov(W))

        end

        @testset "IndexedDistribution" begin
            # FullNormal
            dist = MvNormal(ones(size(A, 1)), A)
            names = "n" .* string.(collect(1:length(dist)))
            id = IndexedDistribution(dist, names)
            chol = cholesky(A)
            @test isequal(sqrtcov(id), sqrtcov(chol))

            # DiagNormal
            Σ = Diagonal(0.1:0.1:0.5)
            dist = MvNormal(ones(size(Σ, 1)), Σ)
            names = "n" .* string.(collect(1:length(dist)))
            id = IndexedDistribution(dist, names)
            @test isequal(sqrtcov(dist), sqrt(Distributions.cov(dist)))
            @test isequal(sqrtcov(dist), sqrt(Σ))
        end
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
        @test X'X ≈ Statistics.cor(data)
        @test sqrtcor(data) ≈ X

        X = sqrtcor(data, w2)
        @test X'X ≈ StatsBase.cor(data, StatsBase.weights(w2))
    end

    @testset "sqrtcor to sqrtcov" begin
        sqrtcor_ = sqrtcor(data, w1)
        std_ = diagm(0 => StatsUtils.std(data, w1))
        sqrtcov_ = sqrtcov(sqrtcor_, std_)
        @test sqrtcov_' * sqrtcov_ ≈ Statistics.cov(data)
    end

    @testset "WeightedResampler" begin
        rng = MersenneTwister(1234)

        @testset "Univariate" begin
            obs = collect(1:12)

            @testset "Equally Weighted" begin
                # Constant analytic weights
                wv = aweights(ones(12))

                s = WeightedResampler(obs, wv)
                X = rand(rng, s, 100000)

                # The mean values of the samples should roughly match the mean of the
                # original observation
                @test isapprox(mean(X), mean(obs); atol=0.01)
            end

            @testset "Linearly Weighted" begin
                # Linearly increasing analytic weights
                wv = aweights(collect(1/12:1/12:1.0))

                s = WeightedResampler(obs, wv)
                X = rand(rng, s, 100000)

                # The mean of the samples should not match the mean of the
                # original observation
                @test !isapprox(mean(X), mean(obs); atol=0.01)

                # 12 should be sampled the most
                @test mode(X) == 12
            end
        end

        @testset "Multivariate" begin
            v = [1.2, 0.7, -0.3, 5.4, -2.8]
            # Define different observations via arbitrary operations on v
            obs = hcat(
                v, reverse(v), sort(v), sin.(v), cos.(v), tan.(v),
                v / 100, v * 2, abs.(v), log.(abs.(v)), v .^ 2, v * 10,
            )

            @testset "Equally Weighted" begin
                # Constant analytic weights
                wv = aweights(ones(12))

                s = WeightedResampler(obs, wv)
                X = rand(rng, s, 100000)

                # The mean values of each variable in the samples should roughly match
                # the means of the original observation
                @test nrmsd(mean(X; dims=2), mean(obs; dims=2)) < 0.001
            end

            @testset "Linearly Weighted" begin
                # Linearly increasing analytic weights
                wv = aweights(collect(0.083:0.083:1.0))

                s = WeightedResampler(obs, wv)
                X = rand(rng, s, 100000)

                # The mean values of each variable of the samples should not match the
                # means of the original observation
                @test nrmsd(mean(X; dims=2), mean(obs; dims=2)) > 0.1

                # v * 10 should be sampled the most
                @test vec(mapslices(mode, X; dims=2)) == v * 10
            end
        end
        @testset "Matrixvariate" begin
            # NOTE: Since we've already testing the sampling behaviour we just want to
            # check that we've implement the Distributions API correctly
            s = WeightedResampler([rand(4, 3) for i in 1:10], aweights(rand(10)))
            X = rand(s)
        end

        @testset "DimensionMismatch" begin
            @test_throws DimensionMismatch WeightedResampler(rand(10), aweights(collect(1:12)))
        end
    end

    @testset "exponential_weights" begin
        λ = 1.1  # must be 0 < λ <= 1
        @test_throws ArgumentError exponential_weights(λ, 10)
        wv1 = exponential_weights(1, 10)
        @test minimum(wv1) == maximum(wv1)  # all have weight 1
        wv2 = exponential_weights(0.5, 10)
        @test first(wv2) < last(wv2)  # weights should be in increasing order
    end  # exponential_weights
end
