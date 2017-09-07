import Base.mean

"""
    mean(x::Var, dim::Int)

Returns the average over the given dimension.
"""
mean(x::Var, dim::Int) = Var(mean(x.data,dim), mean, (x,dim))
mean(x::Node, dim::Int) = Node(mean, x, dim)

function addgrad!(y::Var, ::typeof(mean), x::Var, dim::Int)
    isvoid(x.grad) || ∇mean!(y.grad, x.grad, dim)
end

function ∇mean!{T}(gy::Array{T}, gx::Array{T}, dim::Int)
    g = broadcast(+, x.grad, y.grad)
    broadcast(/, gx, g, size(gx,dim))
end
