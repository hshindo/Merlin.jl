function setup_model()
    T = Float32
    h = 1000
    ls = [Linear(T,784,h), Linear(T,h,h), Linear(T,h,10)]
    (x::Var, y=nothing) -> begin
        x = ls[1](x)
        x = relu(x)
        x = ls[2](x)
        x = relu(x)
        x = ls[3](x)
        y == nothing ? argmax(x.data,1) : crossentropy(y,x)
    end
end

#=
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
=#
