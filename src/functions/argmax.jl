export argmax

"""
    argmax(x::Array, dim::Int)

Returns the maximum elements over the given dimension.
"""
function argmax(x::Array, dim::Int)
    _, index = findmax(x, dim)
    ind2sub(size(x), vec(index))[dim]
end
