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
p = Var(rand(0:10,1,5))
x = Var(rand(Float32,10,5))
y = softmax_crossentropy(p, x)
```
"""
function softmax_crossentropy(p::Var, q::Var; w=nothing)
    if w == nothing
        w = similar(q.data, size(q,1))
        w = Var(fill!(w,1))
    end
    logq = logsoftmax(q.data)
    ydata = softmax_crossentropy(p.data, logq, w.data)
    y = Var(ydata, ∇softmax_crossentropy!, (p,q,logq,w))
    average(y, dims=1)
end

function softmax_crossentropy(p::Vector{Int}, logq::Matrix{T}, w::Vector{T}) where T
    length(p) == size(logq,2) || throw("Length unmatch.")
    y = zeros(T, length(p))
    @inbounds for i = 1:length(y)
        if p[i] > 0
            y[i] = w[p[i]] * -logq[p[i],i]
        end
    end
    y
end

function softmax_crossentropy(p::Matrix{Int}, logq::Matrix{T}) where T
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

function ∇softmax_crossentropy!(y::Var, p::Var, q::Var, logq, w::Var)
    isnothing(q.grad) && return
    ∇softmax_crossentropy!(y.grad, p.data, q.grad, logq, w.data)
end

function ∇softmax_crossentropy!(gy::Vector{T}, p::Vector{Int}, gq::Matrix{T}, logq::Matrix{T}, w::Vector{T}) where T
    @inbounds for j = 1:length(p)
        p[j] <= 0 && continue
        for i = 1:size(logq,1)
            delta = i == p[j] ? T(1) : T(0)
            gq[i,j] += gy[j] * w[p[j]] * (exp(logq[i,j]) - delta)
        end
    end
end

function ∇softmax_crossentropy!(gy::Vector{T}, p::Matrix{T}, gq::Matrix{T}, logq::Matrix{T}) where T
    @inbounds for j = 1:size(p,2)
        for i = 1:size(logq,1)
            gq[i,j] += gy[j] * (exp(logq[i,j]) - p[i,j])
        end
    end
end

@generated function softmax_crossentropy(p::CuVector{Cint}, logq::CuMatrix{T}, w::CuVector{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void softmax_crossentropy($Ct *y, int *p, $Ct *logq, $Ct *w, int size1, int size2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size2) return;
        int qi = (p[idx]-1) + size1 * idx;
        y[idx] = p[idx] > 0 ? -w[p[idx]-1]*logq[qi] : 0;
    }""")
    quote
        length(p) == size(logq,2) || throw("Length unmatch.")
        y = CuArray{T}(length(p))
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), pointer(p), pointer(logq), pointer(w), size(logq,1), size(logq,2))
        y
    end
end

@generated function softmax_crossentropy(p::CuMatrix{T}, logq::CuMatrix{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void softmax_crossentropy($Ct *y, $Ct *p, $Ct *logq, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            y[idx] = -p[idx] * logq[idx];
        }
    }""")
    quote
        size(p) == size(logq) || throw("Length unmatch.")
        y = similar(p)
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), pointer(p), pointer(logq), length(y))
        vec(sum(y,1))
    end
end

@generated function ∇softmax_crossentropy!(gy::CuVector{T}, p::CuVector{Cint}, gq::CuMatrix{T}, logq::CuMatrix{T}, w::CuVector{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void softmax_crossentropy_grad($Ct *gy, int *p, Array<$Ct,2> gq, Array<$Ct,2> logq, $Ct *w) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= logq.length()) return;

        int ndidxs[2];
        logq.ndindex(ndidxs, idx);
        int i = ndidxs[0];
        int j = ndidxs[1];
        if (p[j] > 0) {
            $Ct delta = (i == p[j]-1) ? 1 : 0;
            gq(i,j) += gy[j] * w[p[j]-1] * (exp(logq(i,j)) - delta);
        }
    }""")
    quote
        gdims, bdims = cudims(length(logq))
        $k(gdims, bdims, pointer(gy), pointer(p), gq, logq, pointer(w))
    end
end

@generated function ∇softmax_crossentropy!(gy::CuVector{T}, p::CuMatrix{T}, gq::CuMatrix{T}, logq::CuMatrix{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void softmax_crossentropy_grad($Ct *gy, $Ct *p, $Ct *gq, Array<$Ct,2> logq) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= logq.length()) return;

        int ndidxs[2];
        logq.ndindex(ndidxs, idx);
        int i = ndidxs[0];
        int j = ndidxs[1];
        gq[idx] += gy[j] * (exp(logq[idx]) - p[idx]);
    }""")
    quote
        gdims, bdims = cudims(length(logq))
        $k(gdims, bdims, rawpointer(gy), rawpointer(p), rawpointer(gq), logq)
    end
end
