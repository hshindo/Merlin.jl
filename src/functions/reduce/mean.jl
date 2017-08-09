import Base.mean

"""
    mean(x::Var, dim::Int)

Returns the average over the given dimension.
"""
function mean(x::Var, dim::Int)
    data = mean(x.data, dim)
    Var(data, average, (x,dim))
end

function addgrad!(y::Var, ::typeof(mean), x::Var, dim::Int)
    isvoid(x.grad) && return
    ∇mean!(y.grad, x.grad, dim)
end

function ∇mean!(gy::Array{T}, gx::Array{T}, dim::Int) where {T}
    g = broadcast(+, x.grad, y.grad)
    broadcast(/, gx, g, size(gx,dim))
end
