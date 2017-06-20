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
    data = cat(dim, map(x->x.data,xs)...)
    dim == ndims(xs[1].data) && throw("Invalid cat.")
    for i = 2:length(xs)
        xs[i].batchdims == xs[1].batchdims && continue
        throw("Batchdims mismatch.")
    end
    y = Var(data, xs[1].batchdims, cat, (dim,xs...))
    y.df! = () -> ∇cat!(y.grad, dim, xs...)
    y
end

function ∇cat!(gy::Array{T,N}, dim::Int, xs::Var...) where {T,N}
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
