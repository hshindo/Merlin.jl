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
