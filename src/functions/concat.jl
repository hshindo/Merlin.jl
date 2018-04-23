export concat

"""
    concat(dim::Int, xs::Var...)

Concatenate arrays over the given dimension.

```julia
T = Float32
x1 = Var(rand(T,4,3))
x2 = Var(rand(T,4,5))
y = concat(2, x1, x2)
```
"""
function concat(dim::Int, xs::Var...)
    configure!(xs...)
    y = cat(dim, map(x -> x.data, xs)...)
    Var(y, (concat,dim,xs...))
end
concat(dim::Int, xs::Node...) = Node(concat, dim, xs...)

function addgrad!(y::Var, ::typeof(concat), dim::Int, xs::Var...)
    ∇concat!(y, dim, xs...)
end

function ∇concat!(y::Var, dim::Int, xs::Var...)
    offset = 0
    ysize = Any[Colon() for i=1:ndims(y)]
    for x in xs
        s = size(x, dim)
        if !isvoid(x.grad)
            ysize[dim] = offset+1:offset+s
            gy = view(y.grad, ysize...)
            broadcast!(+, x.grad, x.grad, gy)
        end
        offset += s
    end
end
