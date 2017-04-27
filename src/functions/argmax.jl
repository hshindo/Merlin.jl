export argmax

function argmax(x::Array, dim::Int)
    _, index = findmax(x, dim)
    y = ind2sub(size(x), vec(index))[dim]
    dims = ntuple(i -> i==dim ? 1 : size(x,i), ndims(x))
    reshape(y, dims)
end

"""
    argmax(x::Var, dim::Int)

Returns the maximum elements over the given dimension.
"""
argmax(x::Var, dim::Int) = ArgMax(dim)(x)

type ArgMax
    dim::Int
end

(f::ArgMax)(x::Var) = Var(argmax(x.data,f.dim), f, (x,))
