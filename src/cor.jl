"""
    sqrtcor(data::AbstractMatrix) -> AbstractMatrix

Computes the square root of a pearson correlation matrix for the provided `data`.
Specifically, given an NxM matrix of finite values, the `sqrtcor` method will
return another NxM matrix such that `sqrtcor(data)' * sqrtcor(data) ≈ cor(data)`.

# Usage

```jldoctest
julia> using StatsUtils, Statistics

julia> X = (reshape(1:12, 4, 3) / 12) .^ 2
4×3 Array{Float64,2}:
 0.00694444  0.173611  0.5625
 0.0277778   0.25      0.694444
 0.0625      0.340278  0.840278
 0.111111    0.444444  1.0

julia> sqrtcor_ = StatsUtils.sqrtcor(X)
4×3 Array{Float64,2}:
 -0.572293  -0.634918  -0.648936
 -0.308158  -0.257399  -0.244681
  0.132068   0.18876    0.202128
  0.748383   0.703558   0.691489

julia> sqrtcor_' * sqrtcor_ ≈ Statistics.cor(X)
true
```
"""
sqrtcor(data::AbstractMatrix) = sqrtcor(data, ones(size(data, 1)))

"""
    sqrtcor(data::Abstractmatrix, wv::AbstractVector) -> AbstractMatrix

Computes the square root of a weighted pearson correlation matrix given the provided `data`
and `wv`.

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

julia> StatsUtils.sqrtcor(X, w)
4×3 Array{Float64,2}:
 -0.519615  -0.588602   -0.605302
 -0.489898  -0.466149   -0.459331
 -0.1       -0.0407795  -0.0255711
  0.69282    0.659234    0.649592
```
"""
function sqrtcor(data::AbstractMatrix, wv::AbstractVector)
    @assert size(data, 1) == length(wv)

    sv = _weighted_scale(wv)
    centered = _center(data, wv, 1)

    sqrtcov_ = sqrtcov(centered, wv, sv)
    return sqrtcov_ * Diagonal(1 ./ std(centered, wv, sv))
end

"""
    cor(data::AbstractMatrix, wv::AbstractVector) -> AbstractMatrix

Computes the weighted pearson correlation matrix for the `data` give the
provided `wv`.

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

julia> StatsUtils.cor(X, w)
3×3 Array{Float64,2}:
 1.0       0.995021  0.992158
 0.995021  1.0       0.999675
 0.992158  0.999675  1.0
```
"""
function cor(data::AbstractMatrix, wv::AbstractVector)
    @assert size(data, 1) == length(wv)
    sqrtcor_ = sqrtcor(data, wv)
    return sqrtcor_' * sqrtcor_
end
