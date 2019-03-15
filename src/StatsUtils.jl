module StatsUtils

using LinearAlgebra
using Statistics
using Statistics: sqrt!
using StatsBase

_weighted_scale(wv::AbstractVector; corrected=true) = inv(sum(wv) - corrected)

function _center(data::AbstractMatrix, wv::AbstractVector; dims=1)
    return data .- StatsBase.mean(data, weights(wv); dims=dims)
end

include("cov.jl")
include("cor.jl")
include("std.jl")

end  # module
