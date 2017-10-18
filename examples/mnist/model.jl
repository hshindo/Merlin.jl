type Model
    g
end

function Model{T}(::Type{T}, hsize::Int)
    x = Node()
    h = Linear(T,784,hsize)(x)
    h = relu(h)
    h = Linear(T,hsize,hsize)(h)
    h = relu(h)
    h = Linear(T,hsize,10)(h)
    g = Graph(input=x, output=h)
    Model(g)
end

function (m::Model)(x::Var)
    m.g(x)
end
