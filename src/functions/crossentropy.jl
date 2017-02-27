export crossentropy

function crossentropy{T}(p::Vector{Int32}, logq::Matrix{T})
    length(p) == size(logq,2) || throw(DimensionMismatch())
    y = zeros(logq)
    fill!(y, -10000.0)
    @inbounds @simd for j = 1:length(p)
        p[j] > 0 && (y[p[j],j] = -logq[p[j],j])
    end
    y
end
