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
    ydata = cat(map(x -> x.data, xs)..., dims=dim)
    Var(ydata, (concat,dim,xs...))
end
concat(dim::Int, xs::Node...) = Node(concat, dim, xs...)

addgrad!(y::Var, ::typeof(concat), dim::Int, xs::Var...) = ∇concat!(y, dim, xs...)

function ∇concat!(y::Var, dim::Int, xs::Var...)
    offset = 0
    for x in xs
        s = size(x, dim)
        if !isvoid(x.grad)
            I = ntuple(ndims(y)) do i
                i == dim ? (offset+1:offset+s) : Colon()
            end
            gy = view(y.grad, I...)
            if ndims(y) > ndims(x)
                gy = dropdims(gy, dims=ndims(y))
            end
            addto!(x.grad, gy)
        end
        offset += s
    end
end
