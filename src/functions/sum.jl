import Base.sum

"""
    sum(x::Var, dim::Int)

Returns the sum over the given dimension.
"""
sum(x::Var, dim::Int) = Sum(dim)(x)

type Sum
    dim::Int
end

function (f::Sum)(x::Var)
    y = Var(sum(x.data,f.dim), f, (x,))
    y.df! = function df!()
        isvoid(x.grad) || broadcast!(+, x.grad, x.grad, y.grad)
    end
    y
end
