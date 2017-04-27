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
cat(dim::Int, xs::Var...) = Cat(dim)(xs...)
cat(dim::Int, xs::Vector{Var}) = throw("Use cat(dim, xs...).")

type Cat
    dim::Int
end

function (f::Cat)(xs::Var...)
    y = Var(nothing, f, xs)
    y.data = cat(f.dim, map(x -> x.data, xs)...)
    y.df! = function df!()
        ∇cat!(y.grad, f.dim, xs...)
    end
    y
end

function ∇cat!{T}(gy::Array{T}, dim::Int, xs::Var...)
    offset = 1
    for x in xs
        s = size(x.data, dim)
        if !isvoid(x.grad)
            range = ntuple(ndims(gy)) do i
                i == dim ? (offset:(offset+s-1)) : Colon()
            end
            BLAS.axpy!(T(1), gy[range...], x.grad)
        end
        offset += s
    end
end
