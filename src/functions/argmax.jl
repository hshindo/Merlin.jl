export argmax

"""
    argmax(x::Var, dim::Int)

Returns the maximum elements over the given dimension.
"""
argmax(x::Var, dim::Int) = forward0(argmax, x, dim)

function forward(::typeof(argmax), x::Array, dim::Int)
    _, index = findmax(x, dim)
    y = ind2sub(size(x), vec(index))[dim]
    y, nothing
end

function forward(::typeof(argmax), x::CuArray, dim::Int)
    throw("Not implemented yet.")
end
