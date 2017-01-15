type Model
    f
    ls::Vector
end

function Model{T}(wordembeds::Matrix{T})
    x = Var()
    y = Lookup(wordembeds)(x)
    y = window(y, (500,), pads=(200,), strides=(100,))
    y = Linear(T,500,100)(y)
    y = tanh(y)
    y = window(y, (500,), pads=(200,), strides=(100,))
    y = Linear(T,500,300)(y)

    y = pairwise(y, y)
    f = Graph(y, x)
    ls = [Linear(T,600,300), Linear(T,300,1)]
    Model(f, ls)
end

function (m::Model)(x::Var, y=nothing)
    x = m.f(x)
    n = size(x.data, 2)
    x = reshape(x, size(x.data,1), n*n)
    x = m.ls[1](x)
    x = tanh(x)
    x = m.ls[2](x)
    x = reshape(x, n, n)
    if y == nothing
        argmax(x.data, 1)
    else
        crossentropy(y, x)
    end
end
