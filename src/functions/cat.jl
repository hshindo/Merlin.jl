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
function cat(dim::Int, xs::Vector{Var})
    data = cat(dim, map(getdata,xs))
    y = Var(data, cat, (dim,xs))
    y.df! = () -> begin
        ∇cat!(y.grad, dim, map(getdata,xs), map(getgrad,xs))
    end
    y
end
cat(dim::Int, xs::Var...) = cat(dim, Var[xs...])
cat(dim::Int, xs::Vector{<:Array}) = cat(dim, xs...)

function ∇cat!(gy::Array{T,N}, dim::Int, xs::Vector{<:Array}, gxs::Vector{<:Array}) where {T,N}
    offset = 1
    for i = 1:length(xs)
        x, gx = xs[i], gxs[i]
        s = size(x, dim)
        #if !isconst(x)
            range = ntuple(N) do i
                i == dim ? (offset:(offset+s-1)) : Colon()
            end
            BLAS.axpy!(T(1), gy[range...], gx)
        #end
        offset += s
    end
end
∇cat!(gy::BatchedArray, dim, xs, gxs) = ∇cat!(gy.data, dim, map(getdata,xs), map(getdata,gxs))
