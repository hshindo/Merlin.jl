export concat

"""
    concat(dim::Int, xs::Var...)
    concat(dim::Int, xs::Vector{Var})

Concatenate arrays over the given dimension.

# Example
```julia
x1 = Var(rand(Float32,4,3))
x2 = Var(rand(Float32,4,5))
y = concat(2, x1, x2)
```
"""
function concat(dim::Int, xs::Var...)
    N = ndims(xs[1])
    all(x -> ndims(x) == N, xs) || throw("All `ndims` must be the same: $(map(ndims,xs)).")
    n = maximum(nbatchdims, xs)
    splits = map(xs) do x
        nbatchdims(x) == 1 ? fill(x.data,n) : unsafe_split(x.data,x.batchdims)
    end
    ys = []
    batchdims = Int[]
    for s in zip(splits...)
        y = cat(dim, s...)
        push!(ys, y)
        push!(batchdims, size(y,ndims(y)))
    end
    y = cat(ndims(ys[1]), ys...)
    Var(y, batchdims, concat, (dim,xs...))
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
