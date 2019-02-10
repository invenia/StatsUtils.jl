module StatsUtils

using LinearAlgebra
using Statistics
using Statistics: sqrt!
using StatsBase

_weighted_scale(wv::AbstractVector; corrected=true) = inv(sum(wv) - corrected)

function _center(data::AbstractMatrix, wv::AbstractVector, dim=1)
    return data .- StatsBase.mean(data, weights(wv), dim)
end

include("cov.jl")
include("cor.jl")
include("std.jl")

end  # module
