export softmax_crossentropy

doc"""
    softmax_crossentropy(p, x)

Cross-entropy function between p and ``softmax(x)``.
```math
f(x) = -\sum_{x} p(x) \log q(x)
```
where ``q = softmax(x)``

* p: Var of Vector{Int} or Matrix{Float}
* q: Var of Matrix{Float}

```julia
p = Var(rand(0:10,5))
q = Var(rand(Float32,10,5))
y = softmax_crossentropy(p, x)
```
"""
function softmax_crossentropy(p::Var, x::Var)
    p.batchdims == x.batchdims || throw("Batchdims mismatch: $(p.batchdims), $(x.batchdims)")
    y, logx = softmax_crossentropy(p.data, x.data)
    Var(y, x.batchdims, softmax_crossentropy, (p,x,logx))
end

softmax_crossentropy(p::Node, x::Node; name="") = Node(softmax_crossentropy, p, x, name=name)

function softmax_crossentropy{T}(p::Vector{Int}, x::Matrix{T})
    length(p) == size(x,2) || throw("Length unmatch.")
    logx = logsoftmax(x)
    y = Array{T}(length(p))
    @inbounds for i = 1:length(p)
        y[i] = p[i] > 0 ? -logx[p[i],i] : T(0)
    end
    y, logx
end

function softmax_crossentropy{T}(p::Matrix{T}, x::Matrix{T})
    size(p) == size(x) || throw("Size mismatch.")
    logx = logsoftmax(x)
    y = Array{T}(size(p,2))
    @inbounds for j = 1:size(p,2)
        s = T(0)
        for i = 1:size(p,1)
            s += -p[i,j] * logx[i,j]
        end
        y[j] = s
    end
    y, logx
end

function addgrad!(y::Var, ::typeof(softmax_crossentropy), p::Var, x::Var, logx)
    isvoid(p.grad) || throw("Not implemented yet.")
    isvoid(x.grad) || ∇softmax_crossentropy!(y.grad, p.data, logx, x.grad)
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

function ∇softmax_crossentropy!{T}(gy::Vector{T}, p::Matrix{T}, logx::Matrix{T}, gx::Matrix{T})
    @inbounds for j = 1:size(p,2)
        for i = 1:size(logx,1)
            gx[i,j] += gy[j] * (exp(logx[i,j]) - p[i,j])
        end
    end
end

function ∇softmax_crossentropy!{T}(gy::Matrix{T}, p::Matrix{Int}, q::Matrix{T}, gq::Matrix{T})
    @inbounds for i = 1:length(p)
        p[i] > 0 || continue
        if q[p[i],i] < T(-1e-10) || q[p[i],i] > T(1e-10)
            gq[p[i],i] -= T(1) / q[p[i],i]
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
