import Base.sum

"""
    sum(x::Var, dim::Int)

Compute the sum along the given dimension.
"""
function sum(x::Var, dim::Int)
    x.data == nothing && return Var(nothing, (sum,x,dim))
    dims = ntuple(i -> i==dim ? 1 : size(x,i), ndims(x))
    y = Var(eltype(x), dims, (x,))
    Base.mapreducedim!(identity, +, y.data, x.data)
    y.df = () -> isconst(x) || (x.grad .+= y.grad)
    y
end

#=
"""
    sum(as::Vector{Float64}, xs::Vector{Var})

```math
y=\sum_{i}a_{i}x_{i}
```

where ``a_{i}`` is a scalar and ``x`` is a scholar or n-dimensional array.
The size of each ``x`` might be different. In such a case, the add operation is broadcasted.

### 👉 Example
```julia
as = rand(10)
xs = [Var(rand(Float32,4,3)) for i=1:10]
y = sum(as, xs)
```
"""
function sum(as::Vector{Float64}, xs::Vector{Var})
    maxi, maxlen = 0, 0
    for i = 1:length(xs)
        n = length(xs[i].data)
        n <= maxlen && continue
        maxi = i
        maxlen = n
    end
    y = zeros(xs[maxi].data)
    for i = 1:length(xs)
        y = axpy!(as[i], xs[i].data, y)
    end

    function df(gy)
        for i = 1:length(xs)
            a, x = as[i], xs[i]
            isconst(x) || (x.grad = ∇axpy!(a,x.grad,gy))
        end
    end
    Var(y, xs, sum, df)
end

function axpy!{T}(a::Float64, x::Array{T}, y::Array{T})
    n = length(x)
    for k = 1:n:length(y)
        BLAS.axpy!(n, T(a), pointer(x), 1, pointer(y,k), 1)
    end
    y
end
function axpy!(a::Float64, x::Number, y::Array)
    y .+= a * x
    y
end
axpy!(a::Float64, x::Number, y::Number) = a * x + y

function ∇axpy!{T}(a::Float64, gx::Array{T}, gy::Array{T})
    n = length(gx)
    for k = 1:n:length(gy)
        BLAS.axpy!(n, T(a), pointer(gy,k), 1, pointer(gx), 1)
    end
    gx
end
∇axpy!(a::Float64, gx::Number, gy::Array) = gx + a * sum(gy)
∇axpy!(a::Float64, gx::Number, gy::Number) = gx + a * gy
=#
