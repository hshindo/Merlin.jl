import Base.cat

"""
    cat(dim::Int, xs::Var...)
    cat(dim::Int, xs::Vector{Var})

Concatenate arrays over the given dimension.

```julia
x1 = Var(rand(Float32,4,3))
x2 = Var(rand(Float32,4,5))
y = cat(2, x1, x2)
```
"""
function cat(dim::Int, xs::Vector{Var})
    data = cat(dim, map(x->x.data,xs)...)
    if dim == ndims(xs[1].data)
        batchdims = Int[]
        foreach(xs) do x
            if isvoid(x.batchdims)
                push!(batchdims, size(x.data)[end])
            else
                append!(batchdims, x.batchdims)
            end
        end
    else
        batchdims = xs[1].batchdims
    end
    y = Var(data, batchdims, cat, (dim,xs))
    y.df! = () -> ∇cat!(y.grad, dim, xs)
    y
end
cat(xs::Vector{Var}) = cat(ndims(xs[1].data)), xs)

function ∇cat!(gy::Array{T,N}, dim::Int, xs::Vector{Var}) where {T,N}
    offset = 1
    for x in xs
        s = size(x.data, dim)
        if !isvoid(x.grad)
            range = ntuple(N) do i
                i == dim ? (offset:(offset+s-1)) : Colon()
            end
            BLAS.axpy!(T(1), gy[range...], x.grad)
        end
        offset += s
    end
end
