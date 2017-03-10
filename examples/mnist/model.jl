type Model
    funs::Vector
end

function Model{T}(::Type{T}, hsize::Int)
    funs = [Linear(T,784,hsize), Linear(T,hsize,hsize), Linear(T,hsize,10)]
    Model(funs)
end

function (m::Model)(x::Var, y=nothing)
    x = m.funs[1](x)
    x = relu(x)
    x = m.funs[2](x)
    x = relu(x)
    x = m.funs[3](x)
    y == nothing ? argmax(x.data,1) : crossentropy(y,x)
end

function setup_model{T}(::Type{T}, hsize::Int)
    x = Var()
    y = Var()
    z = var(Linear(T,784,hsize), x)
    z = var(relu, x)
    z = var(Linear(T,hsize,hsize), z)
    z = var(relu, z)
    z = var(Linear(T,hsize,10), z)
    z = var(ifelse, equals(y,nothing), argmax, (x,1), crossentropy, (y,x))
    Graph(z, x, y)
end
