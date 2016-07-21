export crossentropy

doc"""
crossentropy(p::Var, x::Var, dim::Int)

Compute cross-entropy between $p$ and $x$ along the given dimension.

$ f(p,x)=-∑_{i} p_{i} \log x_{i} $

## Arguments
* p: var of `Vector{Int}` or `Matrix{Float}`. $p$ must be normalized.
* x: var of `Matrix{Float}`.

### 👉 Example
```julia
p = Data([1:5;])
x = Data(rand(Float32,10,5))
y = crossentropy(p, x, 1)
```
"""
function crossentropy(p::Var, x::Var, dim::Int)
    hasdata(p,x) || return CrossEntropy(nothing, nothing, [p,x], dim, nothing)
    logx = logsoftmax(x.data, dim)
    y = crossentropy(p.data, logx)
    CrossEntropy(y, nothing, [p,x], dim, logx)
end

type CrossEntropy <: Var
    data
    grad
    tails::Vector
    dim::Int
    logx
end

@compat (v::CrossEntropy)(p::Var, x::Var) = crossentropy(p, x, v.dim)

function backward!(y::CrossEntropy)
    hasgrad(y[2]) || return
    ∇crossentropy!(y[1].data, y.logx, y[2].grad, y.grad)
end

function crossentropy{T}(p::Matrix{T}, logx::Matrix{T})
    y = Array(T, 1, size(p,2))
    for j = 1:size(p,2)
        s = T(0)
        @inbounds @simd for i = 1:size(p,1)
            s += -p[i,j] * logx[i,j]
        end
        y[j] = s
    end
    y
end

function crossentropy{T}(p::Vector{Int}, logx::Matrix{T})
    y = Array(T, 1, length(p))
    @inbounds @simd for j = 1:length(p)
        y[j] = -logx[p[j],j]
    end
    y
end

function ∇crossentropy!{T}(p::Matrix{T}, logx::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
    for j = 1:size(p,2)
        g = gy[j]
        @inbounds @simd for i = 1:size(p,1)
            gx[i,j] += g * (exp(logx[i,j]) - p[i,j])
        end
    end
end

function ∇crossentropy!{T}(p::Vector{Int}, logx::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
    for j = 1:length(p)
        g = gy[j]
        @inbounds @simd for i = 1:size(logx,1)
            delta = ifelse(i == p[j], T(1), T(0))
            gx[i,j] += g * (exp(logx[i,j]) - delta)
        end
    end
end
