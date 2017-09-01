export softmax_crossentropy

"""
    softmax_crossentropy(p::Var, q::Var)

Returns cross-entropy between p and q.
When p[i] == 0, returns 0.

* p: Var of Vector{Int} or Matrix{Float}
* q: Var of Matrix{Float}

```julia
p = Var(rand(0:10,5))
q = Var(rand(Float32,10,5))
y = softmax_crossentropy(p, q)
```
"""
function softmax_crossentropy(p::Var, q::Var)
    p.batchdims == q.batchdims || throw("Batchdims mismatch.")
    logq = logsoftmax(q.data)
    data = softmax_crossentropy(p.data, logq)
    Var(data, q.batchdims, softmax_crossentropy, (p,q), work=logq)
end
softmax_crossentropy(p::Node, q::Node) = Node(softmax_crossentropy, p, q)

function softmax_crossentropy{T}(p::Vector{Int}, logq::Matrix{T})
    length(p) == size(logq,2) || throw("Length unmatch.")
    y = Array{T}(length(p))
    @inbounds for i = 1:length(p)
        y[i] = p[i] > 0 ? -logq[p[i],i] : T(0)
    end
    y
end

function softmax_crossentropy{T}(p::Matrix{T}, logq::Matrix{T})
    size(p) == size(logq) || throw("Size mismatch.")
    y = Array{T}(size(p,2))
    @inbounds for j = 1:size(p,2)
        s = T(0)
        for i = 1:size(p,1)
            s += -p[i,j] * logq[i,j]
        end
        y[j] = s
    end
    y
end

function addgrad!(y::Var, ::typeof(softmax_crossentropy), p::Var, q::Var)
    isvoid(q.grad) && return
    ∇softmax_crossentropy!(y.grad, p.data, y.work, q.grad)
end

#=
@generated function crossentropy{T}(p::CuVector{Int32}, logq::CuMatrix{T})
    f = CuFunction("""
    __global__ void f($T *y, $int *p, Array<$T,2> logq) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < logq.dims[1]) {
            y[idx] = p[idx] > 0 ? -logq(p[idx]-1,idx) : 0;
        }
    }""")
    quote
        length(p) == size(logq,2) || throw(DimensionMismatch())
        y = CuArray{T}(1, length(p))
        $f(y.ptr, p.ptr, logq, dx=length(p))
        y
    end
end
=#

function ∇softmax_crossentropy!{T}(gy::Vector{T}, p::Vector{Int}, logq::Matrix{T}, gq::Matrix{T})
    @inbounds for j = 1:length(p)
        p[j] > 0 || continue
        for i = 1:size(logq,1)
            delta = i == p[j] ? T(1) : T(0)
            gq[i,j] += gy[j] * (exp(logq[i,j]) - delta)
        end
    end
end

function ∇softmax_crossentropy2!{T}(gy::Matrix{T}, p::Matrix{Int}, q::Matrix{T}, gq::Matrix{T})
    @inbounds for i = 1:length(p)
        p[i] > 0 || continue
        if q[p[i],i] < T(-1e-10) || q[p[i],i] > T(1e-10)
            gq[p[i],i] -= T(1) / q[p[i],i]
        end
    end
end

function ∇softmax_crossentropy!{T}(gy::Vector{T}, p::Matrix{T}, logq::Matrix{T}, gq::Matrix{T})
    @inbounds for j = 1:size(p,2)
        for i = 1:size(logq,1)
            gq[i,j] += gy[j] * (exp(logq[i,j]) - p[i,j])
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
