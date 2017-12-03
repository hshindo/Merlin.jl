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
    aa = map(xs) do x
        nbatchdims(x) == 1 ? fill(x.data,n) : unsafe_split(x.data,x.batchdims)
    end
    ys = []
    for p in zip(aa...)
        push!(ys, cat(dim,p...))
    end
    cat(dim, ys...)

    if dim == N
        batchdims = Int[]
        for x in xs
             append!(batchdims, x.batchdims)
        end
        y = cat(dim, map(x -> x.data, xs)...)
        Var(y, batchdims, concat, (dim,xs...))
    elseif length(xs) == 2
        nbatchdims(xs[1]) == 1


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
