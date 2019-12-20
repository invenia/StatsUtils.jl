"""
    exponential_weights(λ::Number, n::Integer) -> AnalyticWeights

Return `n` exponentially increasing weights, with  `λ` specifying the ratio between
successive data points. That is, ``λ^t`` for  ``0 < λ ≤ 1``, ``t = n-1, n-2, …, 0``.

Note: this can be contrasted with [`StatsBase.eweights`](https://juliastats.org/StatsBase.jl/stable/weights/#Exponential-weights:-eweights-1).
Which is similar in purpose, but with a harder to interpret λ parameter.
```jldoctest; setup = :(using StatsUtils: exponential_weights)
julia> λ = 0.75;

julia> n = 5;

julia> exponential_weights(λ, n)
5-element StatsBase.AnalyticWeights{Float64,Float64,Array{Float64,1}}:
 0.31640625
 0.421875
 0.5625
 0.75
 1.0
```
"""
function exponential_weights(λ::Number, n::Integer)
    # This is the definition the Research team prefer. Could use StatsBase if it adopts this.
    # AnalyticWeights specifies the bias correction term we want for `std` and `cov`.
    0 < λ ≤ 1 || throw(ArgumentError("Must be 0 < λ ≤ 1. Got λ=$λ"))
    return AnalyticWeights([λ^t for t in (n - 1):-1:0])
end
