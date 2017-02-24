export crossentropy

"""
    crossentropy(p::Var, q::Var)

Returns cross-entropy between p and q.
When p[i] == 0, returns 0.

* p: Var of Vector{Int} or Matrix{Float}
* q: Var of Matrix{Float}

```julia
p = Var(rand(0:10,5))
q = Var(rand(Float32,10,5))
y = crossentropy(p, q)
```
"""
function crossentropy(p::Var, q::Var)
    logq = logsoftmax(q.data)
    y = crossentropy(p.data, logq)
    df(v::Var) = isvoid(v[2].grad) || ∇crossentropy!(v.grad, p.data, logq, v[2].grad)
end

function crossentropy{T}(p::Matrix{T}, logq::Matrix{T})
    y = Array(T, 1, size(p,2))
    for j = 1:size(p,2)
        s = T(0)
        @inbounds @simd for i = 1:size(p,1)
            s += -p[i,j] * logq[i,j]
        end
        y[j] = s
    end
    y
end

function crossentropy{T}(p::Vector{Int}, logq::Matrix{T})
    length(p) == size(logq,2) || throw(DimensionMismatch())
    y = Array(T, 1, length(p))
    @inbounds @simd for j = 1:length(p)
        y[j] = p[j] > 0 ? -logq[p[j],j] : T(0)
    end
    y
end

function ∇crossentropy!{T}(gy::Matrix{T}, p::Matrix{T}, logq::Matrix{T}, gq::Matrix{T})
    for j = 1:size(p,2)
        g = gy[j]
        @inbounds @simd for i = 1:size(p,1)
            gq[i,j] += g * (exp(logq[i,j]) - p[i,j])
        end
    end
end

function ∇crossentropy!{T}(gy::Matrix{T}, p::Vector{Int}, logq::Matrix{T}, gq::Matrix{T})
    for j = 1:length(p)
        g = gy[j]
        @inbounds @simd for i = 1:size(logq,1)
            if p[j] > 0
                delta = ifelse(i == p[j], T(1), T(0))
                gq[i,j] += g * (exp(logq[i,j]) - delta)
            end
        end
    end
end
