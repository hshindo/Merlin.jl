import Base.cat

"""
    cat(dim::Int, xs::Var...)

Concatenate arrays over the given dimension.

# Example
```julia
x1 = Var(rand(Float32,4,3))
x2 = Var(rand(Float32,4,5))
y = cat(2, x1, x2)
```
"""
function cat(dim::Int, xs::Var...)
    y = cat(dim, map(x -> x.data, xs)...)
    Var(y, cat, (dim,xs...))
end
cat(dim::Int, xs::Node...; name="") = Node(cat, (dim,xs...), name)

function addgrad!(y::Var, ::typeof(cat), dim::Int, xs::Var...)
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
