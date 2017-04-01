export argmax

function argmax(x::Array, dim::Int)
    _, index = findmax(x, dim)
    ind2sub(size(x), vec(index))[dim]
end

"""
    argmax(x::Var, dim::Int)

Returns the maximum elements over the given dimension.
"""
argmax(x::Var, dim::Int) = forward0(argmax, x, dim)

forward(::typeof(argmax), x::UniArray, dim::Int) = argmax(x,dim), nothing
