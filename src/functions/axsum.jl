export axsum

"""
    axsum

```math
y = \sum_{i} a_{i} \cdot x_{i}
```

where ``a_{i}`` is a scalar and ``x`` is scholar or vector.
Every operation is broadcasted.
"""
function axsum(as::Vector{Float64}, xs::Vector{Var})
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
            hasgrad(x) && (x.grad = ∇axpy!(a,x.grad,gy))
        end
    end
    Var(y, xs, axsum, df)
end

function axpy!{T}(a::Float64, x::UniArray{T}, y::UniArray{T})
    n = length(x)
    for k = 1:n:length(y)
        BLAS.axpy!(n, T(a), pointer(x), 1, pointer(y,k), 1)
    end
    y
end
function axpy!(a::Float64, x::Number, y::UniArray)
    y .+= a * x
    y
end
axpy!(a::Float64, x::Number, y::Number) = a * x + y

function ∇axpy!{T}(a::Float64, gx::UniArray{T}, gy::UniArray{T})
    n = length(gx)
    for k = 1:n:length(gy)
        BLAS.axpy!(n, T(a), pointer(gy,k), 1, pointer(gx), 1)
    end
    gx
end
∇axpy!(a::Float64, gx::Number, gy::Array) = gx + a * sum(gy)
∇axpy!(a::Float64, gx::Number, gy::Number) = gx + a * gy
