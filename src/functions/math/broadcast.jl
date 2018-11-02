import Base.Broadcast: broadcasted

"""
    .+(x1::Var, x2::Var)
"""
function broadcasted(::typeof(+), x1::Var, x2::Var)
    ydata = x1.data .+ x2.data
    Var(ydata, ∇broadcasted!, (+,x1,x2))
end
broadcasted(::typeof(+), x1::Node, x2) = Node(broadcasted, (+,x1,x2))
broadcasted(::typeof(+), x1, x2::Node) = Node(broadcasted, (+,x1,x2))

function ∇broadcasted!(y::Var, ::typeof(+), x1::Var, x2::Var)
    isnothing(x1.grad) || ∇broadcast_plus!(y.grad, x1.grad)
    isnothing(x2.grad) || ∇broadcast_plus!(y.grad, x2.grad)
end

function ∇broadcast_plus!(gy::UniArray{T}, gx::UniArray{T}) where T
    if length(gy) == length(gx)
        addto!(gx, gy)
    else
        dims = ()
        for i = 1:ndims(gy)
            if size(gx,i) == 1 && size(gy,i) > 1
                dims = tuple(dims..., i)
            end
        end
        addto!(gx, sum(gy,dims=dims))
    end
end

"""
    .-(x1::Var, x2::Var)
"""
function broadcasted(::typeof(-), x1::Var, x2::Var)
    ydata = x1.data .- x2.data
    Var(ydata, ∇broadcasted!, (-,x1,x2))
end
broadcasted(::typeof(-), x1::Node, x2) = Node(broadcasted, (-,x1,x2))
broadcasted(::typeof(-), x1, x2::Node) = Node(broadcasted, (-,x1,x2))

function ∇broadcasted!(y::Var, ::typeof(-), x1::Var, x2::Var)
    isnothing(x1.grad) || ∇broadcast_plus!(y.grad, x1.grad)
    isnothing(x2.grad) || ∇broadcast_minus!(y.grad, x2.grad)
end

function ∇broadcast_minus!(gy::UniArray{T}, gx::UniArray{T}) where T
    if length(gy) == length(gx)
        axpy!(T(-1), gy, gx)
    else
        dims = ()
        for i = 1:ndims(gy)
            if size(gx,i) == 1 && size(gy,i) > 1
                dims = tuple(dims..., i)
            end
        end
        axpy!(T(-1), sum(gy,dims=dims), gx)
    end
end

"""
    .*(x1::Var, x2::Var)
"""
function broadcasted(::typeof(*), x1::Var, x2::Var)
    ydata = x1.data .* x2.data
    Var(ydata, ∇broadcasted!, (*,x1,x2))
end
broadcasted(::typeof(*), x1::Node, x2) = Node(broadcasted, (*,x1,x2))
broadcasted(::typeof(*), x1, x2::Node) = Node(broadcasted, (*,x1,x2))

function ∇broadcasted!(y::Var, ::typeof(*), x1::Var, x2::Var)
    isnothing(x1.grad) || ∇dottimes!(y.grad, x2.data, x1.grad)
    isnothing(x2.grad) || ∇dottimes!(y.grad, x1.data, x2.grad)
end

function ∇dottimes!(gy::UniArray{T}, x2::UniArray{T}, gx1::UniArray{T}) where T
    g = gy .* x2
    if length(gx1) == length(g)
        addto!(gx1, g)
    else
        dims = ()
        for i = 1:ndims(gy)
            if size(gx1,i) == 1 && size(g,i) > 1
                dims = tuple(dims..., i)
            end
        end
        addto!(gx1, sum(g,dims=dims))
    end
end
