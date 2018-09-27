export crossentropy

doc"""
    crossentropy(p, q)

Cross-entropy function between p and q.

```math
f(x) = -\sum_{x} p(x) \log q(x)
```

* p::Var: `Var` of Vector{Int} or Matrix{Float}. If p is `Vector{Int}` and p[i] == 0, returns 0.
* q::Var: `Var` of Matrix{Float}

```julia
p = Var(rand(0:10,5))
q = softmax(Var(rand(Float32,10,5)))
y = crossentropy(p, q)
```
"""
function crossentropy(p::Var, q::Var)
    configure!(p, q)
    Var(crossentropy(p.data,q.data), (crossentropy,p,q))
end

function crossentropy(p::Vector{Int}, q::Matrix{T}) where T
    y = Array{T}(length(p))
    @inbounds for i = 1:length(p)
        y[i] = p[i] > 0 ? -log(q[p[i],i]) : T(0)
    end
    y
end

function crossentropy(p::Vector{T}, q::Vector{T}) where T
    @assert length(p) == length(q)
    y = T[0]
    @inbounds for i = 1:length(p)
        y[1] -= p[i] * log(q[i])
    end
    y
end

function addgrad!(y::Var, ::typeof(crossentropy), p::Var, q::Var)
    @assert isvoid(p.grad)
    isvoid(q.grad) && return
    ∇crossentropy!(y.grad, p.data, q.data, q.grad)
end

function ∇crossentropy!(gy::Vector{T}, p::Vector{I}, q::Matrix{T}, gq::Matrix{T}) where {I<:Integer,T}
    @inbounds for i = 1:length(p)
        if p[i] > 0
            gq[p[i],i] -= gy[i] / q[p[i],i]
        end
    end
end

function ∇crossentropy!(gy::Vector{T}, p::Vector{T}, q::Vector{T}, gq::Vector{T}) where T
    @assert length(gy) == 1
    @inbounds for i = 1:length(p)
        gq[i] -= gy[1] * p[i] / q[i]
    end
end
