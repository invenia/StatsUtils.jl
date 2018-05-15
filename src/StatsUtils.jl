module StatsUtils

using Compat
using Compat.LinearAlgebra

_weighted_scale(weights::AbstractVector) = inv(sum(weights) - 1)
_center(data::AbstractMatrix, dim=1) = data .- Compat.mean(data, dims=dim)

"""
    std(data::AbstractMatrix, weights::AbstractVector) -> AbstractVector

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
 0.0440728
 0.109226
 0.174619
```
"""
function std(data::AbstractMatrix, weights::AbstractVector)
    @assert size(data, 1) == length(weights)
    return std(_center(data), weights, _weighted_scale(weights))
end

"""
    std(centered::AbstractMatrix, weights::AbstractVector, sv::Real) -> AbstractVector

Computes the remainder of the weighted standard deviation along the first axis
using already centered data and a scale value (`sv`).

NOTE: you probably don't want to call this directly.
"""
function std(centered::AbstractMatrix, weights::AbstractVector, sv::Real)
    return Base.sqrt!(Compat.rmul!(vec(Compat.sum(weights .* centered .^ 2, dims=1)), sv))
end

"""
    sqrtcov(data::AbstractMatrix) -> AbstractMatrix

Computes the square root of a covariance matrix for the provided `data`.
Specifically, given an NxM matrix of finite values, the `sqrtcov` method will
return another NxM matrix such that `sqrtcov(data)' * sqrtcov(data) ≈ cov(data)`

# Usage

```jldoctest
julia> using StatsUtils

julia> X = (reshape(1:12, 4, 3) / 12) .^ 2
4×3 Array{Float64,2}:
 0.00694444  0.173611  0.5625
 0.0277778   0.25      0.694444
 0.0625      0.340278  0.840278
 0.111111    0.444444  1.0

julia> sqrtcov_ = StatsUtils.sqrtcov(X)
4×3 Array{Float64,2}:
 -0.0260609   -0.0741735  -0.122286
 -0.0140328   -0.0300703  -0.0461078
  0.00601407   0.0220516   0.0380891
  0.0340797    0.0821922   0.130305

julia> sqrtcov_' * sqrtcov_ ≈ cov(X)
true
```
"""
sqrtcov(data::AbstractMatrix) = sqrtcov(data, ones(size(data, 1)))

"""
    sqrtcov(data::AbstractMatrix, weights::AbstractVector) -> AbstractMatrix

Computes the square root of a weighted covariance matrix for the provided `data`
and `weights`.

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

julia> StatsUtils.sqrtcov(X, w)
4×3 Array{Float64,2}:
 -0.0150463   -0.0428241  -0.0706019
 -0.0114577   -0.0245523  -0.0376469
  0.00601407   0.0220516   0.0380891
  0.0393519    0.0949074   0.150463
```
"""
function sqrtcov(data::AbstractMatrix, weights::AbstractVector)
    @assert size(data, 1) == length(weights)

    sv = _weighted_scale(weights)
    centered = _center(data, 1)

    return sqrtcov(centered, weights, sv)
end

"""
    sqrtcov(centered::AbstractMatrix, weights::AbstraactVector, sv::Real) -> AbstractMatrix

Computes the remainder of the weighted covariance matrix along the first axis
using the alread centered data and scale value (`sv`).

NOTE: you probably don't want to call this directly.
"""
function sqrtcov(centered::AbstractMatrix, weights::AbstractVector, sv::Real)
    return sqrt.(weights) .* centered .* sqrt(sv)
end

"""
    sqrtcov(sqrtcor_::AbstractMatrix, stds::AbstractMatrix) -> AbstractMatrix

Computes a MxM upper triangular square root covariance matrix from a
NxM square root pearson correlation matrix (`sqrtcor_`) and an MxM diagonal
matrix of standard deviations (`stds`).

NOTE: This is just a utility method for a QR matrix decomposition
(ie: `qr(sqrtcor_ * stds)[2]`)

# Usage

```jldoctest
julia> using StatsUtils

julia> X = (reshape(1:12, 4, 3) / 12) .^ 2
4×3 Array{Float64,2}:
 0.00694444  0.173611  0.5625
 0.0277778   0.25      0.694444
 0.0625      0.340278  0.840278
 0.111111    0.444444  1.0

julia> statscov_ = StatsUtils.sqrtcov(StatsUtils.sqrtcor(X), diagm(0=>vec(std(X, 1))))
3×3 Array{Float64,2}:
 0.0455378   0.116139    0.18674
 0.0        -0.0126295  -0.025259
 0.0         0.0         1.21443e-16

julia> statscov_' * statscov_ ≈ cov(X)
true
```
"""
function sqrtcov(sqrtcor_::AbstractMatrix, stds::AbstractMatrix)
    @assert size(sqrtcor_, 2) == size(stds, 1)
    @assert isdiag(stds)
    _, sqrtcor_ = qr(sqrtcor_ * stds)
    return sqrtcor_
end

"""
    cov(data::AbstractMatrix, weights::AbstractVector) -> AbstractMatrix

Computes the weighted covariance matrix for the `data` give the provided
`weights`.

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

julia> StatsUtils.cov(X, w)
3×3 Array{Float64,2}:
 0.00194241  0.00479306  0.00764371
 0.00479306  0.0119304   0.0190678
 0.00764371  0.0190678   0.0304918
```
"""
function cov(data::AbstractMatrix, weights::AbstractVector)
    @assert size(data, 1) == length(weights)
    sqrtcov_ = sqrtcov(data, weights)
    return sqrtcov_' * sqrtcov_
end

"""
    sqrtcor(data::AbstractMatrix) -> AbstractMatrix

Computes the square root of a pearson correlation matrix for the provided `data`.
Specifically, given an NxM matrix of finite values, the `sqrtcor` method will
return another NxM matrix such that `sqrtcor(data)' * sqrtcor(data) ≈ cor(data)`.

# Usage

```jldoctest
julia> using StatsUtils

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

julia> sqrtcor_' * sqrtcor_ ≈ cor(X)
true
```
"""
sqrtcor(data::AbstractMatrix) = sqrtcor(data, ones(size(data, 1)))

"""
    sqrtcor(data::Abstractmatrix, weights::AbstractVector) -> AbstractMatrix

Computes the square root of a weighted pearson correlation matrix given the provided `data`
and `weights`.

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
 -0.341397  -0.392067  -0.404319
 -0.259973  -0.224784  -0.215594
  0.136458   0.201889   0.218127
  0.892884   0.868905   0.861664

```
"""
function sqrtcor(data::AbstractMatrix, weights::AbstractVector)
    @assert size(data, 1) == length(weights)

    sv = _weighted_scale(weights)
    centered = _center(data, 1)

    sqrtcov_ = sqrtcov(centered, weights, sv)
    return sqrtcov_ * diagm(0 => 1 ./ std(centered, weights, sv))
end

"""
    cov(data::AbstractMatrix, weights::AbstractVector) -> AbstractMatrix

Computes the weighted pearson correlation matrix for the `data` give the
provided `weights`.

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
 1.0       0.995669  0.993213
 0.995669  1.0       0.999725
 0.993213  0.999725  1.0
```
"""
function cor(data::AbstractMatrix, weights::AbstractVector)
    @assert size(data, 1) == length(weights)
    sqrtcor_ = sqrtcor(data, weights)
    return sqrtcor_' * sqrtcor_
end

end  # module
