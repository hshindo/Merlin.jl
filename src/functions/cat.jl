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
    y = Var(nothing, cat, (dim,xs...))
    any(x -> isvoid(x.data), xs) && return y

    y.data = cat(dim, map(x->x.data,xs)...)
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
