type Model
    l1
    l2
    l3
end

function Model()
    T = Float32
    h = 1000
    l1 = Linear(T, 784, h)
    l2 = Linear(T, h, h)
    l3 = Linear(T, h, 10)
    Model(l1, l2, l3)
end

function (m::Model)(x::Var, y=nothing)
    h = 1000 # hidden vector size
    x = x |> m.l1 |> relu |> m.l2 |> relu |> m.l3
    y == nothing && return argmax(x.data, 1)
    crossentropy(y, x)
end
