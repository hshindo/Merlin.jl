type Model
    l1
    l2
    l3
    ls::Vector
end

function Model{T}(wordembeds::Matrix{T})
    l1 = Lookup(wordembeds)
    l2 = GatedLinear(T,300,300)
    l3 = GatedLinear(T,900,300)
    ls = [Linear(T,600,300), Linear(T,300,1)]
    Model(l1, l2, l3, ls)
end

type Model2
    f
    ls::Vector
end

function Model2{T}(wordembeds::Matrix{T})
    x = Var()
    y = Lookup(wordembeds)(x)
    y = window(y, (300,), pads=(100,), strides=(100,))
    y = GatedLinear(T,300,300)(y)
    y = window(y, (900,), pads=(300,), strides=(300,))
    y = GatedLinear(T,900,300)(y)
    y = window(y, (900,), pads=(300,), strides=(300,))
    y = GatedLinear(T,900,300)(y)
    y = window(y, (900,), pads=(300,), strides=(300,))
    y = GatedLinear(T,900,300)(y)
    y = window(y, (900,), pads=(300,), strides=(300,))
    y = GatedLinear(T,900,300)(y)
    y = window(y, (900,), pads=(300,), strides=(300,))
    y = GatedLinear(T,900,300)(y)
    #y = window(y, (900,), pads=(300,), strides=(300,))
    #y = GatedLinear(T,900,300)(y)
    #y = window(y, (900,), pads=(300,), strides=(300,))
    #y = GatedLinear(T,900,300)(y)

    y = pairwise(y, y)
    f = Graph(y, x)
    ls = [Linear(T,600,300), Linear(T,300,1)]
    Model2(f, ls)
end

function (m::Model)(x::Var, y=nothing)
    n = length(x.data)
    x = m.l1(x)
    x = window(x, (300,), pads=(100,), strides=(100,))
    x = m.l2(x)
    for i = 1:length(n)
        y == nothing || (x = dropout(x, 0.5))
        x = window(x, (900,), pads=(300,), strides=(300,))
        x = m.l3(x)
    end
    x = pairwise(x, x)
    n = size(x.data, 2)
    x = reshape(x, size(x.data,1), n*n)
    x = m.ls[1](x)
    x = tanh(x)
    x = m.ls[2](x)
    x = reshape(x, n, n)
    y == nothing ? argmax(x.data,1) : crossentropy(y, x)
end

function (m::Model2)(x::Var, y=nothing)
    x = m.f(x)
    n = size(x.data, 2)
    x = reshape(x, size(x.data,1), n*n)
    x = m.ls[1](x)
    x = tanh(x)
    x = m.ls[2](x)
    x = reshape(x, n, n)
    y == nothing ? argmax(x.data,1) : crossentropy(y, x)
end
