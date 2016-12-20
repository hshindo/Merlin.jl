export dropout

"""
    dropout(x::Var, rate::Float64, istrain::Bool)
"""
function dropout(x::Var, rate::Float64)
    x.data == nothing && return Var(nothing, dropout, (x,rate))
    dropout(typeof(x.data), x, rate)
end

function dropout{T<:Array}(::Type{T}, x::Var, rate::Float64)
    rx = rand(eltype(x.data), length(x.data))
    y = dropout(x.data, rate, rx)
    df(gy) = x.grad == nothing || ∇dropout!(gy, x.grad, rate, rx)
    Var(y, df, (x,))
end

function dropout{T<:CuArray}(::Type{T}, x::Var, rate::Float64)
    y, work = CUDNN.dropout(x.data, rate)
    df(gy::CuArray) = x.grad == nothing || ∇dropout!(gy, x.grad, rate, work::DropoutWork)
    Var(y, df, (x,))
end

function dropout{T}(x::Array{T}, rate::Float64, rx::Array{T})
    scale = T(1.0 / (1.0-rate))
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = ifelse(rx[i] <= T(rate), T(0), scale*x[i])
    end
    y
end

function ∇dropout!{T}(gy::Array{T}, gx::Array{T}, rate::Float64, rx::Array{T})
    scale = T(1.0 / (1.0-rate))
    @inbounds @simd for i = 1:length(gx)
        gx[i] += ifelse(rx[i] <= T(rate), T(0), scale*gy[i])
    end
    gx
end
