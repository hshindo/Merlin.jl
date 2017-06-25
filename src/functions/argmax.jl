export argmax

"""
    argmax(x::Var, dim::Int)

Returns the maximum elements over the given dimension.
"""
function argmax(x::Var, dim::Int)
    y = Var(nothing, argmyax, (x,))
    isvoid(x.data) && return y

    y.data = argmax(x.data, dim)
    y
end

function argmax(x::Array, dim::Int)
    _, index = findmax(x, dim)
    y = ind2sub(size(x), vec(index))[dim]
    dims = ntuple(i -> i==dim ? 1 : size(x,i), ndims(x))
    reshape(y, dims)
end
