export crossentropy

"""
    crossentropy(p::Var, q::Var)

Returns cross-entropy between p and q.
* p: Var of Vector{Int} or Matrix{Float}
* q: Var of Matrix{Float}

## 👉 Example
```julia
p = Var([1:5;])
q = Var(rand(Float32,10,5))
y = crossentropy(p, q)
```
"""
function crossentropy(p::Var, q::Var)
    (p.data == nothing || q.data == nothing) && return Var(nothing, crossentropy, (p,q))
    logq = logsoftmax(q.data)
    y = crossentropy(p.data, logq)
    df(gy) = q.grad == nothing || ∇crossentropy!(gy, p.data, logq, q.grad)
    Var(y, df, (q,))
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
        y[j] = -logq[p[j],j]
    end
    y
end

function crossentropy{T}(p::CuVector{Cint}, logq::CuArray{T})
    length(p) == size(logq,2) || throw(DimensionMismatch())
    y = CuArray{T}(1, length(p))
    t = CUDA.ctype(T)
    f = @nvrtc """
    $(CUDA.array_h)
    __global__ void f(Array<int,1> p, Array<$t,2> logq, Array<$t,2> y) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < p.length()) {
            y[idx] = -logq(p[idx]-1, idx);
        }
    } """
    f(p, logq, y, dx=length(y))
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
            delta = ifelse(i == p[j], T(1), T(0))
            gq[i,j] += g * (exp(logq[i,j]) - delta)
        end
    end
end

function ∇crossentropy!{T}(gy::CuMatrix{T}, p::CuVector{Cint}, logq::CuMatrix{T}, gx::CuMatrix{T})
    length(p) == size(logq,2) || throw(DimensionMismatch())
    y = CuArray{T}(1, length(p))
    t = CUDA.ctype(T)
    f = @nvrtc """
    $(CUDA.array_h)
    __global__ void f(Array<int,1> p, Array<$t,2> logq, Array<$t,2> y) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < p.length()) {
            y[idx] = -logq(p[idx]-1, idx);
        }
    } """
    f(p, logq, y, dx=length(y))
    y
end
