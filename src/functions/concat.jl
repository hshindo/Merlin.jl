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
function concat(dim::Int, xs::Vector{Var})
    configure!(xs...)
    y = cat(dim, map(x -> x.data, xs)...)
    Var(y, (concat,dim,xs))
end
concat(dim::Int, xs::Var...) = concat(dim, [xs...])
concat(dim::Int, xs::Node...) = Node(concat, dim, xs...)

function addgrad!(y::Var, ::typeof(concat), dim::Int, xs::Vector{Var})
    ∇concat!(y, dim, xs)
end

function ∇concat!(y::Var, dim::Int, xs::Vector{Var})
    offset = 0
    for x in xs
        s = size(x, dim)
        if !isvoid(x.grad)
            I = ntuple(ndims(y)) do i
                i == dim ? (offset+1:offset+s) : Colon()
            end
            gy = view(y.grad, I...)
            ndims(y) > ndims(x) && (gy = squeeze(gy,ndims(y)))
            add!(x.grad, gy)
        end
        offset += s
    end
end
