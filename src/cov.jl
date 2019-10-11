"""
    sqrtcov(X::MvNormal) -> AbstractMatrix
    sqrtcov(X::IndexedDistribution) -> AbstractMatrix
    sqrtcov(Σ::PDMat) -> AbstractMatrix
    sqrtcov(Σ::PSDMat) -> AbstractMatrix
    sqrtcov(Σ::PDiagMat) -> AbstractMatrix
"""
sqrtcov(Σ::PDiagMat) = sqrt.(Matrix(Σ))
# No public API for assessing `chol`; see https://github.com/JuliaStats/PDMats.jl/issues/88
sqrtcov(Σ::Union{PDMat, PSDMat}) = sqrtcov(Σ.chol)
# Do not use`Distributions.cov` as it returns a `Matrix`; we want original type of `Σ`.
sqrtcov(X::MvNormal) = sqrtcov(X.Σ)
sqrtcov(id::IndexedDistribution) = sqrtcov(parent(id))

"""
    sqrtcov(F::Cholesky) -> AbstractMatrix
    sqrtcov(F::CholeskyPivoted) -> AbstractMatrix
    sqrtcov(F::LinearAlgebra.QRCompactWY) -> AbstractMatrix

Handle the extraction of the square root from a given covariance Factorization.
Needed because implementation details such as pivoting mean additional handling of the
Factorization needed to be taken into account.


```jldoctest
julia> using StatsUtils, LinearAlgebra

julia> X = (reshape(1:9, 3, 3) / 12) .^ 2
3×3 Array{Float64,2}:
 0.00694444  0.111111  0.340278
 0.0277778   0.173611  0.444444
 0.0625      0.25      0.5625

julia> U = Matrix(UpperTriangular(X))
3×3 Array{Float64,2}:
 0.00694444  0.111111  0.340278
 0.0         0.173611  0.444444
 0.0         0.0       0.5625

julia> A = U' * U
3×3 Array{Float64,2}:
 4.82253e-5   0.000771605  0.00236304
 0.000771605  0.0424865    0.114969
 0.00236304   0.114969     0.629726

julia> C = cholesky(A, Val(true))
LinearAlgebra.CholeskyPivoted{Float64,Array{Float64,2}}
U factor with rank 3:
3×3 UpperTriangular{Float64,Array{Float64,2}}:
 0.793553  0.144879  0.0029778
  ⋅        0.146617  0.00232022
  ⋅         ⋅        0.00582877
permutation:
3-element Array{Int64,1}:
 3
 2
 1

julia> StatsUtils.sqrtcov(C)' * StatsUtils.sqrtcov(C) ≈ A
true
```
"""
sqrtcov(F::LinearAlgebra.QRCompactWY) = F.R
sqrtcov(F::Cholesky) = F.U
function sqrtcov(F::CholeskyPivoted)
    ip = invperm(F.p)
    return F.U[ip, ip]
end

"""
    sqrtcov(data::AbstractMatrix) -> AbstractMatrix

Computes the square root of a covariance matrix for the provided `data`.
Specifically, given an NxM matrix of finite values, the `sqrtcov` method will
return another NxM matrix such that `sqrtcov(data)' * sqrtcov(data) ≈ cov(data)`

# Usage

```jldoctest
julia> using StatsUtils, Statistics

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

julia> sqrtcov_' * sqrtcov_ ≈ Statistics.cov(X)
true
```
"""
sqrtcov(data::AbstractMatrix) = sqrtcov(data, ones(size(data, 1)))

"""
    sqrtcov(data::AbstractMatrix, wv::AbstractVector) -> AbstractMatrix

Computes the square root of a weighted covariance matrix for the provided `data`
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

julia> StatsUtils.sqrtcov(X, w)
4×3 Array{Float64,2}:
 -0.0208333   -0.0578704   -0.0949074
 -0.0196419   -0.045831    -0.0720201
 -0.00400938  -0.00400938  -0.00400938
  0.0277778    0.0648148    0.101852
```
"""
function sqrtcov(data::AbstractMatrix, wv::AbstractVector; corrected=true)
    @assert size(data, 1) == length(wv)

    sv = _weighted_scale(wv; corrected=corrected)
    centered = _center(data, wv; dims=1)

    return sqrtcov(centered, wv, sv)
end

"""
    sqrtcov(centered::AbstractMatrix, weights::AbstraactVector, sv::Real) -> AbstractMatrix

Computes the remainder of the weighted covariance matrix along the first axis
using the alread centered data and scale value (`sv`).

NOTE: you probably don't want to call this directly.
"""
function sqrtcov(centered::AbstractMatrix, wv::AbstractVector, sv::Real)
    return sqrt.(wv) .* centered .* sqrt(sv)
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
julia> using StatsUtils, Statistics, LinearAlgebra

julia> X = (reshape(1:12, 4, 3) / 12) .^ 2
4×3 Array{Float64,2}:
 0.00694444  0.173611  0.5625
 0.0277778   0.25      0.694444
 0.0625      0.340278  0.840278
 0.111111    0.444444  1.0

julia> statscov_ = StatsUtils.sqrtcov(StatsUtils.sqrtcor(X), Diagonal(vec(Statistics.std(X; dims=1))))
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
    cov(data::AbstractMatrix, wv::AbstractVector) -> AbstractMatrix

Computes the weighted covariance matrix for the `data` give the provided
`wv`.

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
 0.00160751  0.00392233  0.00623714
 0.00392233  0.0096665   0.0154107
 0.00623714  0.0154107   0.0245842
```
"""
function cov(data::AbstractMatrix, wv::AbstractVector; corrected=true)
    @assert size(data, 1) == length(wv)
    sqrtcov_ = sqrtcov(data, wv; corrected=corrected)
    return sqrtcov_' * sqrtcov_
end
