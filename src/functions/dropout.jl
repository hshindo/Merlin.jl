export dropout

doc"""
    dropout(x::Var, rate::Float64)

Drops elements randomly with probability ``rate`` and scales the other elements by factor ``1 / (1 - rate)``.
"""
function dropout(x::Var, rate::Float64)
    rate == 0.0 && return x
    y = Var(nothing, (x,))
    dropout!(y, x.data, rate)
end
dropout(x::Node, rate::Float64; name="") = Node(dropout, (x,rate), name)
dropout(x::Array, rate::Float64) = x

function dropout!(out, x::Array{T}, rate::Float64) where T
    rx = rand(T, length(x.data))
    scale = T(1 / (1-rate))
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = rx[i] <= rate ? T(0) : scale*x[i]
    end

    out.data = y
    out.∇! = function ∇!()
        isvoid(out[1].grad) || ∇dropout!(out.grad, out[1].grad, T(rate), rx)
    end
    out
end

function ∇dropout!(gy::Array{T}, gx::Array{T}, rate::T, rx::Vector{T}) where T
    scale = T(1 / (1-rate))
    @inbounds for i = 1:length(gx)
        gx[i] += ifelse(rx[i] <= rate, T(0), scale*gy[i])
    end
end
