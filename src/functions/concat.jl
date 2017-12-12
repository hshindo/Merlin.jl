export concat

"""
    concat(dim::Int, xs::Var...)

Concatenate arrays over the given dimension.

# Example
```julia
x1 = Var(rand(Float32,4,3))
x2 = Var(rand(Float32,4,5))
y = concat(2, x1, x2)
```
"""
function concat(dim::Int, xs::Vector{Var})
    if any(x -> isvoid(x.data), xs)
        y = nothing
    else
        y = cat(dim, map(x -> x.data, xs)...)
    end
    Var(y, (concat,dim,xs))
end
concat(dim::Int, xs::Var...) = concat(dim, [xs...])

function addgrad!(y::Var, ::typeof(concat), dim::Int, xs::Vector{Var})
    T, N = eltype(y), ndims(y)
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
