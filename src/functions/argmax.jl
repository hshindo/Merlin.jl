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
argmax(x::Var, dim::Int) = forward0(argmax, x, dim)

forward(::typeof(argmax), x::UniArray, dim::Int) = argmax(x,dim), nothing
