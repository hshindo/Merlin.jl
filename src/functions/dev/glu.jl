export GLU

type GLU <: Functor
    insize::Int
    outsize::Int
    linear::Linear
end

"""
    GLU(T::Type, insize::Int, outsize::Int)

Gated Linear Unit.
"""
function GLU{T}(::Type{T}, insize::Int, outsize::Int)
    linear = Linear(T, insize, 2outsize)
    GLU(insize, outsize, linear)
end

function (f::GLU)(x::Var)
    h = f.linear(x)
    n = f.outsize
    y = h[1:n] .* sigmoid(h[n+1:2n])
    y
end
