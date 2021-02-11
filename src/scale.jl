"""
    scale(dist::Union{MvNormal, GenericMvTDist}) -> AbstractPDMat
    scale(dist::Normal) -> Real

extract the scale parameter from a distribution. It preserves the original `AbstractPDMat`
type for multivariate distributions.
"""
function scale(dist::Union{MvNormal, GenericMvTDist})
    return dist.Σ
end

scale(dist::Normal) = dist.σ

scale(dist::IndexedDistribution) = scale(parent(dist))
