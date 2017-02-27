export crossentropy

function crossentropy(p::Var, logq::Var)
    logq[]
end

function crossentropy{T}(p::Vector{Int32}, logq::Matrix{T})
    length(p) == size(logq,2) || throw(DimensionMismatch())
    y = Array{T}(1, length(p))
    @inbounds @simd for j = 1:length(p)
        y[j] = p[j] > 0 ? -logq[p[j],j] : T(0)
    end
    y
end
