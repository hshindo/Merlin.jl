export argmax

doc"""
    argmax(x, dim::Int)

Returns the maximum elements over the given dimension.
"""
function argmax(x::Array, dim::Int)
    _, index = findmax(x, dim)
    y = ind2sub(size(x), vec(index))[dim]
    dims = ntuple(i -> i==dim ? 1 : size(x,i), ndims(x))
    reshape(y, dims)
end
