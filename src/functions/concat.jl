export concat

"""
    concat(dim::Int, xs::Var...)
    concat(dim::Int, xs::Vector{Var})

Concatenate arrays over the given dimension.

```julia
x1 = Var(rand(Float32,4,3))
x2 = Var(rand(Float32,4,5))
y = concat(2, x1, x2)
```
"""
function concat(dim::Int, xs::Var...)
    @assert all(x -> ndims(x) == ndims(xs[1]), xs)
    if dim == ndims(xs[1])
        batchdims = Int[]
        for x in xs
             append!(batchdims, x.batchdims)
        end
        y = cat(dim, map(x -> x.data, xs)...)
        Var(y, batchdims, concat, (dim,xs...))
    else
        batchdims1 = xs[1].batchdims
        all(x -> x.batchdims == batchdims1, xs) || throw("Batchdims are not the same: $(map(x -> x.batchdims, xs))")
        y = cat(dim, map(x -> x.data, xs)...)
        Var(y, batchdims1, concat, (dim,xs...))
    end

end

concat(dim::Int, xs::Node...; name="") = Node(concat, (dim,xs...), name)

function addgrad!(y::Var, ::typeof(concat), dim::Int, xs::Var...)
    T, N = eltype(y), ndims(y)
    offset = 1
    for x in xs
        s = size(x, dim)
        if !isvoid(x.grad)
            range = ntuple(N) do i
                i == dim ? (offset:(offset+s-1)) : Colon()
            end
            BLAS.axpy!(T(1), y.grad[range...], x.grad)
        end
        offset += s
    end
end
