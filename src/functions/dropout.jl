export dropout

"""
    dropout(x::Var, rate::Float64)
"""
function dropout{T,N}(x::Var{Array{T,N}}, rate::Float64)
    rx = rand(T, length(x.data))
    scale = T(1 / (1-rate))
    y = similar(x.data)
    @inbounds @simd for i = 1:length(x.data)
        y[i] = ifelse(rx[i] <= T(rate), T(0), scale*x.data[i])
    end
    df(gy) = isvoid(x.grad) || ∇dropout!(gy, x.grad, rate, rx)
    Var(y, df, (x,))
end
dropout(x::Var{Void}, rate::Float64) = Var(Void(), dropout, (x,rate))

function ∇dropout!{T}(gy::Array{T}, gx::Array{T}, rate::Float64, rx::Array{T})
    scale = T(1/(1-rate))
    @inbounds @simd for i = 1:length(gx)
        gx[i] += ifelse(rx[i] <= T(rate), T(0), scale*gy[i])
    end
    gx
end
