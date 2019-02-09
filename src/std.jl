#=
NOTE: The std functions below are simply so we can use an arbitrary vector as a weight
vector with the bias correction used in the legacy system. Once the weight types are easier
to use and we are comfortable using the bias correction code in StatsBase then we should be
able to phase out most of these methods.
=#

"""
    std(data::AbstractMatrix, wv::AbstractVector) -> AbstractVector

Compute the weighted standard deviation along the first axis.

# Usage

```jldoctest
julia> using StatsUtils

julia> X = (reshape(1:12, 4, 3) / 12) .^ 2
4×3 Array{Float64,2}:
 0.00694444  0.173611  0.5625
 0.0277778   0.25      0.694444
 0.0625      0.340278  0.840278
 0.111111    0.444444  1.0

julia> w = collect(1:4)
4-element Array{Int64,1}:
 1
 2
 3
 4

julia> StatsUtils.std(X, w)
3-element Array{Float64,1}:
 0.040093768693724004
 0.09831833602590415
 0.15679346501737126
```
"""
function std(data::AbstractMatrix, wv::AbstractVector; corrected=true)
    @assert size(data, 1) == length(wv)
    return std(_center(data, wv), wv, _weighted_scale(wv; corrected=corrected))
end

"""
    std(centered::AbstractMatrix, wv::AbstractVector, sv::Real) -> AbstractVector

Computes the remainder of the weighted standard deviation along the first axis
using already centered data and a scale value (`sv`).

NOTE: you probably don't want to call this directly.
"""
function std(centered::AbstractMatrix, wv::AbstractVector, sv::Real)
    return sqrt!(rmul!(vec(sum(wv .* centered .^ 2, dims=1)), sv))
end
