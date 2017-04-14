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
crossentropy(p::Var, q::Var) = forward0(crossentropy, p, q)

function forward(::typeof(crossentropy), p::Array{Int}, q::Array)
    logq = logsoftmax(q)
    y = crossentropy(p, logq)
    backward!(gy, gp, gq) = isvoid(gq) || ∇crossentropy!(gy, p, logq, gq)
    y, backward!
end

function forward(::typeof(crossentropy), p::CuVector{Int32}, q::CuMatrix)
    logq = logsoftmax(q)
    y = crossentropy(p, logq)
    backward!(gy, gp, gq) = isvoid(gq) || ∇crossentropy!(gy, p, logq, gq)
    y, backward!
end

function crossentropy{T}(p::Matrix{Int}, logq::Matrix{T})
    size(p,1) == 1 || throw(DimensionMismatch("size(p,1) != 1"))
    size(p,2) == size(logq,2) || throw(DimensionMismatch("size of p: $(size(p)), size of logq: $(size(logq))"))
    y = Array{T}(1, length(p))
    @inbounds @simd for j = 1:length(p)
        y[j] = p[j] > 0 ? -logq[p[j],j] : T(0)
    end
    y
end

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

function ∇crossentropy!{T}(gy::Matrix{T}, p::Matrix{Int}, logq::Matrix{T}, gq::Matrix{T})
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

@generated function ∇crossentropy!{T}(gy::CuMatrix{T}, p::CuVector{Int32}, logq::CuMatrix{T}, gq::CuMatrix{T})
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

function ∇crossentropy!{T}(gy::Matrix{T}, p::Matrix{T}, logq::Matrix{T}, gq::Matrix{T})
    for j = 1:size(p,2)
        g = gy[j]
        @inbounds @simd for i = 1:size(p,1)
            gq[i,j] += g * (exp(logq[i,j]) - p[i,j])
        end
    end
end
