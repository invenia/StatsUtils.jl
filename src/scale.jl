"""
    scale(dist::Union{MvNormal, GenericMvTDist}) -> AbstractPDMat

extract the scale parameter from a distribution. It preserves the original `AbstractPDMat`
type for multivariate distributions.
"""
function scale(dist::Union{MvNormal, GenericMvTDist})
    return dist.Σ
end

scale(dist::IndexedDistribution) = scale(parent(dist))
