type Clipping
    minval::Float64
    maxval::Float64
end

function (opt::Clipping){T}(x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] = min(max(gx[i], opt.minval), opt.maxval)
    end
end
