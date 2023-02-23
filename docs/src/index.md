# StatsUtils

StatsUtils.jl contains various statistics which may be incorporated into [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl) at a later date.

## Features

* `sqrtcov(X)`: get square root of covariance matrix such `sqrtcov(X)' * sqrtcov(X) ≈ cov(X)`
* `sqrtcor(X)`: get square root of pearson correlation matrix such `sqrtcor(X)' * sqrtcor(X) ≈ cor(X)`
* `cov(X, w)` & `cor(X, w)` for getting weight covariance and correlation.
* `sqrtcov(sqrtcor_, stds)`: for getting a upper trangular square root of a covariance matrix using the square root of a pearson correlation matrix and a diagonal matrix of standard deviations from the base source data.
