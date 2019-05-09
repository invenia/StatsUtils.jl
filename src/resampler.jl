struct Resampler{F<:VariateForm, S<:ValueSupport} <: Sampleable{F, S}
    obs::AbstractArray
    wv::AbstractWeights
end

"""
    Resampler(obs::AbstractArray, wv::AbstractWeights)

A Resampler is a subtype of Distributions.Sampleable which randomly selects
observations from the raw input data (`obs`) based on the weights (`wv`) provided.

This type supports univariate, multivariate and matrixvariate forms, so `obs` can
be a vector of values, matrix of values or a vector of matrices.
"""
function Resampler(obs::T, wv::AbstractWeights) where T<:AbstractArray
    F = _variate_form(T)
    S = _value_support(eltype(T))

    _validate(obs, wv)
    Resampler{F, S}(obs, wv)
end

_variate_form(::Type{<:AbstractVector}) = Univariate
_variate_form(::Type{<:AbstractMatrix}) = Multivariate
_variate_form(::Type{<:AbstractVector{<:AbstractMatrix}}) = Matrixvariate

_value_support(::Type{Int}) = Discrete
_value_support(::Type{Float64}) = Continuous
_value_support(T::Type{<:AbstractMatrix}) = _value_support(eltype(T))

_validate(obs::AbstractVector, wv::AbstractWeights) = _validate(length(obs), length(wv))
_validate(obs::AbstractMatrix, wv::AbstractWeights) = _validate(size(obs, 2), length(wv))

function _validate(nobs::Int, nwv::Int)
    nobs == nwv || throw(DimensionMismatch(
        "Length of the weights vector ($nwv) must match the number of observations ($nobs)."
    ))
end

Base.length(s::Resampler{Multivariate}) = size(s.obs, 1)

Base.rand(s::Resampler) = rand(Random.GLOBAL_RNG, s)
function Distributions._rand!(s::Resampler{Multivariate}, x::AbstractVector{T}) where T<:Real
    _rand!(Random.GLOBAL_RNG, s, x)
end

function Base.rand(
    rng::AbstractRNG, s::Resampler{F}
) where F<:Union{Univariate, Matrixvariate}
    i = sample(rng, s.wv)
    return s.obs[i]
end

function Distributions._rand!(
    rng::AbstractRNG, s::Resampler{Multivariate}, x::AbstractVector{T}
) where T<:Real
    j = sample(rng, s.wv)
    for i in 1:length(s)
        @inbounds x[i] = s.obs[i, j]
    end
end
