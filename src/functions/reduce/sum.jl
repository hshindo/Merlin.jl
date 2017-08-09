import Base.sum

"""
    sum(x::Var, dim::Int)

Returns the sum over the given dimension.
"""
function sum(x::Var, dim::Int)
    y = Var(nothing, sum, (x,dim))
    y.data = sum(x.data, dim)
    y.df! = () -> begin
        isvoid(x.grad) || broadcast!(+, x.grad, x.grad, y.grad)
    end
    y
end
