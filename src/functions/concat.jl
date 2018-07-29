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
    Var(y, (concat,dim,xs))
end
concat(dim::Int, xs::Node...) = Node(concat, dim, xs...)

function concat(dim::Int, xs::Vector{Var})
    @assert dim == ndims(xs[1])
    if iscontigious(xs)
        s = (Base.front(size(xs[1]))..., sum(size,xs))
        y = unsafe_wrap(typeof(xs[1]), pointer(xs[1].data), s)
    else
        y = cat(dim, map(x -> x.data, xs)...)
    end
    Var(y, (concat,dim,xs))
end

function iscontigious(xs::Vector{Var})
    p = pointer(xs[1].data)
    for x in xs
        p == pointer(x.data) || return false
        p = pointer(x.data, length(x)+1)
    end
    true
end

function addgrad!(y::Var, ::typeof(concat), dim::Int, xs)
    ∇concat!(y, dim, xs)
end

function ∇concat!(y::Var, dim::Int, xs)
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
