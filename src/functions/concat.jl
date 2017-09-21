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
    N = ndims(xs[1])
    batchdims1 = xs[1].batchdims
    samesize = all(x -> x.batchdims == batchdims1, xs)
    if dim == N
        throw("Not implemented yet.")
        batchdims = Int[]
        foreach(x -> append!(batchdims,x.batchdims), xs)
    else
        samesize || throw("Batchdims are not the same.")
        batchdims = batchdims1
    end
    y = cat(dim, map(x -> x.data, xs)...)
    Var(y, batchdims, concat, (dim,xs...))
end

concat(dim::Int, xs::Node...; name="concat") = Node(concat, dim, xs..., name=name)

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
