type Model
    g::Graph
end

function Model{T}(::Type{T}, hsize::Int)
    x = Var()
    y = Var()
    h = Linear(T,784,hsize)(x)
    h = relu(h)
    h = Linear(T,hsize,hsize)(h)
    h = relu(h)
    h = Linear(T,hsize,10)(h)
    g = Graph((x,), (h,))
    Model(g)
end

function (m::Model)(x::Var, y=nothing)
    z = m.g(x)
    y == nothing ? argmax(z.data,1) : crossentropy(y,z)
end
