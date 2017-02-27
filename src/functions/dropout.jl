export dropout

"""
    dropout(x::Var, rate::Float64)

This is inteded to be used only for training.
For testing, omit the dropout function.
"""
dropout(x::Var, rate::Float64) = forward0(dropout, x, rate)

function forward(::typeof(dropout), x::Array, rate::Float64)
    rx = rand(T, length(x))
    scale = T(1 / (1-rate))
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = ifelse(rx[i] <= T(rate), T(0), scale*x[i])
    end
    backward!(gy, gx) = isvoid(gx) || ∇dropout!(gy, gx, rate, rx)
    y, backward!
end

function ∇dropout!{T}(gy::Array{T}, gx::Array{T}, rate::Float64, rx::Array{T})
    scale = T(1/(1-rate))
    @inbounds @simd for i = 1:length(gx)
        gx[i] += ifelse(rx[i] <= T(rate), T(0), scale*gy[i])
    end
end
