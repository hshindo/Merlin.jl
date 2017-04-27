import Base.reshape

"""
    reshape(x::Var, dims::Int...)
"""
reshape(x::Var, dims::Tuple) = Reshape(dims)(x)
reshape(x::Var, dims::Int...) = reshape(x, dims)

type Reshape
    dims::Tuple
end

function (f::Reshape)(x::Var)
    y = Var(reshape(x.data,f.dims), f, (x,))
    y.df! = function df!()
        T = eltype(x.data)
        isvoid(x.grad) || BLAS.axpy!(T(1), y.grad, x.grad)
    end
    y
end
