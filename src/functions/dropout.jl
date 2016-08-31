export dropout

"""
    dropout(x::Var, ratio::Float64, istrain::Bool)
"""
function dropout(x::Var, ratio::Float64, istrain::Bool)
    istrain || return x
    if typeof(x.data) <: Array
        rx = rand(eltype(x.data), length(x.data))
        y = dropout(x.data, ratio, rx)
        df(gy) = hasgrad(x) && ∇dropout!(ratio, rx, x.grad, gy)
    else
        throw("Not implemented yet.")
    end
    Var(y, [x], dropout, df)
end

function dropout{T}(x::Array{T}, ratio::Float64, rx::Array{T})
    scale = T(1.0 / (1.0-ratio))
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = ifelse(rx[i] <= T(ratio), T(0), scale*x[i])
    end
    y
end

dropout{T}(x::CuArray, ratio::Float64, rx::CuArray{T}) = dropout(x, ratio)

function ∇dropout!{T}(ratio::Float64, rx::Array{T}, gx::Array{T}, gy::Array{T})
    scale = T(1.0 / (1.0-ratio))
    @inbounds @simd for i = 1:length(gx)
        gx[i] += ifelse(rx[i] <= T(ratio), T(0), scale*gy[i])
    end
    gx
end

function ∇dropout!{T}(ratio::Float64, states, statessize, reserve, reservesize,
    gx::CuArray{T}, gy::CuArray{T})

    ∇dropout!(gy, ratio, states, statessize, reserve, reservesize, gx)
end
