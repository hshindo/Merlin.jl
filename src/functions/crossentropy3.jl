export crossentropy

function forward{T}(::typeof(crossentropy), p::Vector{Int32}, logq::Matrix{T})
    length(p) == size(logq,2) || throw(DimensionMismatch())
    y = zeros(logq)
    @inbounds @simd for j = 1:length(p)
        p[j] > 0 && (y[p[j],j] = -1)
    end
    y
end
