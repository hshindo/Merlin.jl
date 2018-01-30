export dropout

doc"""
    dropout(x::Var, droprate::Float64)

Drops elements randomly with probability ``droprate`` and scales the other elements by factor ``1 / (1 - droprate)``.
"""
function dropout(x::Var, droprate::Float64)
    droprate == 0.0 && return x
    CONFIG.train || return x
    y, r = dropout(x.data, droprate)
    Var(y, (dropout,x,droprate,r))
end

dropout(x::Node, droprate) = Node(dropout, x, droprate)

function dropout(x::Array{T}, droprate::Float64) where T
    r = rand(T, length(x))
    scale = T(1 / (1-droprate))
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = r[i] <= droprate ? T(0) : scale*x[i]
    end
    y, r
end

function dropout(x::CuArray, droprate)
    CUDNN.dropout(x, droprate)
end

function addgrad!(y::Var, ::typeof(dropout), x::Var, droprate::Float64, r)
    isvoid(x.grad) && return
    ∇dropout!(y.grad, x.grad, droprate, r)
end

function ∇dropout!(gy::Array{T}, gx::Array{T}, droprate::Float64, r::Vector{T}) where T
    scale = T(1 / (1-droprate))
    @inbounds for i = 1:length(gx)
        gx[i] += r[i] <= droprate ? T(0) : scale*gy[i]
    end
end

function ∇dropout!(gy::CuArray, gx, droprate, r)
    CUDNN.∇dropout!(gy, gx, droprate, r)
end
