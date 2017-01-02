export dropout

"""
    dropout(x::Var, rate::Float64)
"""
function dropout(x::Var, rate::Float64)
    isvoid(x.data) && return Var(nothing, dropout, (x,rate))
    iscuda(x.data) && return CUDA.dropout(x, rate)

    T = eltype(x.data)
    rx = rand(T, length(x.data))
    scale = T(1 / (1-rate))
    y = similar(x.data)
    @inbounds @simd for i = 1:length(x.data)
        y[i] = ifelse(rx[i] <= T(rate), T(0), scale*x.data[i])
    end
    df(gy) = isvoid(x.grad) || ∇dropout!(gy, x.grad, rate, rx)
    Var(y, df, (x,))
end

function ∇dropout!{T}(gy::Array{T}, gx::Array{T}, rate::Float64, rx::Array{T})
    scale = T(1/(1-rate))
    @inbounds @simd for i = 1:length(gx)
        gx[i] += ifelse(rx[i] <= T(rate), T(0), scale*gy[i])
    end
    gx
end
