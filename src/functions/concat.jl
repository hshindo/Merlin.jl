export concat

"""
    concat(dim::Int, xs::Var...)

Concatenate arrays over the given dimension.

```julia
T = Float32
x1 = Var(rand(T,4,3))
x2 = Var(rand(T,4,5))
y = concat(2, x1, x2)
```
"""
function concat(dim::Int, xs::Var...)
    y = cat(dim, map(x -> x.data, xs)...)
    Var(y, (concat,dim,xs...))
end
concat(dim::Int, xs::Vector{Var}) = concat(dim, xs...)
concat(dim::Int, xs::Node...; name="") = Node(concat, (dim,xs...), name)

function addgrad!(y::Var, ::typeof(concat), dim::Int, xs::Var...)
    gxs = map(x -> x.grad, xs)
    ∇concat!(y.grad, dim, gxs...)
end

function ∇concat!(gy::Array{T,N}, dim::Int, gxs...) where {T,N}
    offset = 1
    for gx in gxs
        s = size(gx, dim)
        range = ntuple(N) do i
            i == dim ? (offset:(offset+s-1)) : Colon()
        end
        BLAS.axpy!(T(1), gy[range...], gx)
        offset += s
    end
end

function ∇concat!(gy::CuArray, dim::Int, gxs::CuArray...)
    LibCUDA.∇concat!(gy, dim, gxs...)
end
