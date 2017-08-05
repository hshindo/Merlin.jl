import Base.cat

"""
    cat(dim::Int, xs::Var...)

Concatenate arrays over the given dimension.

```julia
x1 = Var(rand(Float32,4,3))
x2 = Var(rand(Float32,4,5))
y = cat(2, x1, x2)
```
"""
function cat(dim::Int, xs::Var...)
    data = cat(dim, map(x -> x.data, xs)...)
    N = ndims(xs[1].data)
    if dim == N
        batchdims = cat(1, map(x -> x.batchdims, xs)...)
    else
        all(x -> x.batchdims == xs[1].batchdims, xs) || throw("invalid batchdims")
        batchdims = xs[1].batchdims
    end
    Var(data, batchdims, cat, (dim,xs...))
end
cat(dim::Int, xs::Node...) = Node(dim, xs...)

function addgrad!(y::Var, ::typeof(cat), dim::Int, xs::Var...)
    T, N = eltype(y.data), ndims(y.data)
    offset = 1
    for x in xs
        s = size(x.data, dim)
        if !isvoid(x.grad)
            range = ntuple(N) do i
                i == dim ? (offset:(offset+s-1)) : Colon()
            end
            BLAS.axpy!(T(1), y.grad[range...], x.grad)
        end
        offset += s
    end
end
