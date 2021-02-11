module StatsUtils

using Distributions
using Distributions: GenericMvTDist
using IndexedDistributions
using LinearAlgebra
using PDMats
using PDMatsExtras
using Random
using Statistics
using Statistics: sqrt!
using StatsBase

export WeightedResampler, exponential_weights

_weighted_scale(wv::AbstractVector; corrected=true) = inv(sum(wv) - corrected)

function _center(data::AbstractMatrix, wv::AbstractVector; dims=1)
    return data .- StatsBase.mean(data, weights(wv); dims=dims)
end

include("weights.jl")
include("cov.jl")
include("cor.jl")
include("scale.jl")
include("std.jl")
include("resampler.jl")

end  # module
