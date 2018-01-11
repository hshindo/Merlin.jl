export dropout

doc"""
    dropout(x::Var, rate::Float64, train::Bool)

Drops elements randomly with probability ``rate`` and scales the other elements by factor ``1 / (1 - rate)``.
"""
function dropout(x::Var, droprate::Float64)
    rate == 0.0 && return x
    y, work = dropout(x.data, droprate)
    Var(y, (dropout,x,droprate), work=work)
end
dropout(x::Node, rate::Node; name="") = Node(dropout, (x,rate), name)

function dropout(x::Array{T}, droprate::Float64) where T
    work = rand(T, length(x.data))
    scale = T(1 / (1-droprate))
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = work[i] <= droprate ? T(0) : scale*x[i]
    end
    y, work
end

function addgrad!(y::Var, ::typeof(dropout), x::Var, droprate::Float64)
    isvoid(x.grad) && return
    ∇dropout!(y.grad, x.grad, droprate, y.work)
end

function ∇dropout!(gy::Array{T}, gx::Array{T}, droprate::Float64, work::Vector{T}) where T
    scale = T(1 / (1-droprate))
    @inbounds for i = 1:length(gx)
        gx[i] += work[i] <= droprate ? T(0) : scale*gy[i]
    end
end
