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
    y = cat(dim, map(x -> x.data, xs)...)
    Var(y, (concat,dim,xs...))
end
concat(dim::Int, xs::Vector{Var}) = concat(dim, xs...)
concat(dim::Int, xs::Node...) = Node(concat, dim, xs...)

function addgrad!(y::Var, ::typeof(concat), dim::Int, xs::Var...)
    ∇concat!(y.grad, dim, xs...)
end

function ∇concat!(gy::Array{T,N}, dim::Int, xs::Var...) where {T,N}
    offset = 0
    ysize = Any[Colon() for i=1:N]
    for x in xs
        s = size(x, dim)
        if !isvoid(x.grad)
            ysize[dim] = offset+1:offset+s
            BLAS.axpy!(T(1), gy[ysize...], x.grad)
        end
        offset += s
    end
end
