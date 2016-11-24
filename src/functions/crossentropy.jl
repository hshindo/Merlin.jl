export crossentropy

"""
    crossentropy(p::Var, x::Var; normalize=true)

Computes cross-entropy between p and x.
* p: Var of Vector{Int} or Matrix{Float}

If normalize=true, x is normalized.

## 👉 Example
```julia
p = Var([1:5;])
x = Var(rand(Float32,10,5))
y = crossentropy(p, x)
```
"""
function crossentropy(p::Var, x::Var; normalize=true)
    normalize == false && throw("Not implemented yet.")

    x.data == nothing && return Var(nothing, crossentropy, (p,x))
    logpx = logsoftmax(x.data)
    y = crossentropy(p.data, logpx)
    df(gy) = isconst(x) || ∇crossentropy!(gy, p.data, logpx, x.grad)
    Var(y, crossentropy, (p,x), df)
end
crossentropy(p, x::Var; normalize=true) = crossentropy(Var(p), x, normalize=normalize)

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

function ∇crossentropy!{T}(gy::Matrix{T}, p::Matrix{T}, logx::Matrix{T}, gx::Matrix{T})
    for j = 1:size(p,2)
        g = gy[j]
        @inbounds @simd for i = 1:size(p,1)
            gx[i,j] += g * (exp(logx[i,j]) - p[i,j])
        end
    end
end

function ∇crossentropy!{T}(gy::Matrix{T}, p::Vector{Int}, logx::Matrix{T}, gx::Matrix{T})
    for j = 1:length(p)
        g = gy[j]
        @inbounds @simd for i = 1:size(logx,1)
            delta = ifelse(i == p[j], T(1), T(0))
            gx[i,j] += g * (exp(logx[i,j]) - delta)
        end
    end
end
