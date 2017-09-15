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
    p.batchdims == q.batchdims || throw("Batchdims mismatch: $(p.batchdims) : $(q.batchdims)")
    Var(crossentropy(p.data,q.data), p.batchdims, crossentropy, (p,q))
end

crossentropy(p::Node, q::Node; name="crossentropy") = Node(crossentropy, p, q, name=name)

function crossentropy{T}(p::Vector{Int}, q::Matrix{T})
    y = Array{T}(length(p))
    for i = 1:length(p)
        y[i] = p[i] > 0 ? -log(q[p[i],i]) : T(0)
    end
    y
end

function addgrad!(y::Var, ::typeof(crossentropy), p::Var, q::Var)
    isvoid(q.grad) || ∇crossentropy!(y.grad, p.data, q.data, q.grad)
end

function ∇crossentropy!{T}(gy::Vector{T}, p::Vector{Int}, q::Matrix{T}, gq::Matrix{T})
    @inbounds for i = 1:length(p)
        if p[i] > 0
            gq[p[i],i] -= gy[i] / q[p[i],i]
        end
    end
end

#=
@generated function ∇softmax_crossentropy!{T}(gy::CuMatrix{T}, p::CuVector{Int32}, logq::CuMatrix{T}, gq::CuMatrix{T})
    f = CuFunction("""
    __global__ void f($T *gy, $T *p, Array<$T,2> logq, Array<$T,2> gq) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= logq.length()) return;

        int subs[2];
        logq.idx2sub(subs);
        int i = subs[0];
        int j = subs[1];
        if (p[j] > 0) {
            $T delta = (i == p[j]-1) ? 1 : 0;
            gq(i,j) += gy[j] * (exp(logq(i,j)) - delta);
        }
    }""")
    quote
        $f(gy.ptr, p.ptr, logq, gq, dx=length(logq))
    end
end
=#
