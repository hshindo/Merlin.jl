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
    configure!(p, x)
    logx = logsoftmax(x.data)
    y = softmax_crossentropy(p.data, logx)
    Var(y, (softmax_crossentropy,p,x,logx))
end

function softmax_crossentropy(p::Vector{I}, logx::Matrix{T}) where {T,I<:Integer}
    length(p) == size(logx,2) || throw("Length unmatch.")
    y = Array{T}(length(p))
    @inbounds for i = 1:length(p)
        y[i] = p[i] > 0 ? -logx[p[i],i] : T(0)
    end
    y
end

function softmax_crossentropy(p::Matrix{T}, logx::Matrix{T}) where T
    size(p) == size(logx) || throw("Size mismatch.")
    y = Array{T}(size(p,2))
    @inbounds for j = 1:size(p,2)
        s = T(0)
        for i = 1:size(p,1)
            s += -p[i,j] * logx[i,j]
        end
        y[j] = s
    end
    y
end

function addgrad!(y::Var, ::typeof(softmax_crossentropy), p::Var, x::Var, work)
    @assert isvoid(p.grad)
    isvoid(x.grad) && return
    ∇softmax_crossentropy!(y.grad, p.data, x.grad, work)
end

function ∇softmax_crossentropy!(gy::Vector{T}, p::Vector{I}, gx::Matrix{T}, logx::Matrix{T}) where {T,I<:Integer}
    @inbounds for j = 1:length(p)
        p[j] <= 0 && continue
        for i = 1:size(logx,1)
            delta = i == p[j] ? T(1) : T(0)
            gx[i,j] += gy[j] * (exp(logx[i,j]) - delta)
        end
    end
end

function ∇softmax_crossentropy!(gy::Vector{T}, p::Matrix{T}, gx::Matrix{T}, logx::Matrix{T}) where T
    @inbounds for j = 1:size(p,2)
        for i = 1:size(logx,1)
            gx[i,j] += gy[j] * (exp(logx[i,j]) - p[i,j])
        end
    end
end

@generated function softmax_crossentropy(p::CuVector{Cint}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void softmax_crossentropy($Ct *y, int *p, Array<$Ct,2> logx, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            y[idx] = p[idx] > 0 ? -logx(p[idx]-1,idx) : 0;
        }
    }""")
    quote
        length(p) == size(logx,2) || throw("Length unmatch.")
        y = CuArray{T}(length(p))
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), pointer(p), logx, length(y))
        y
    end
end

@generated function softmax_crossentropy(p::CuMatrix{T}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void softmax_crossentropy($Ct *y, $Ct *p, $Ct *logx, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            y[idx] = -p[idx] * logx[idx];
        }
    }""")
    quote
        size(p) == size(logx) || throw("Length unmatch.")
        y = similar(p)
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), pointer(p), pointer(logx), length(y))
        vec(sum(y,1))
    end
end

@generated function ∇softmax_crossentropy!(gy::CuVector{T}, p::CuVector{Cint}, gx::CuMatrix{T}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void softmax_crossentropy_grad($Ct *gy, int *p, Array<$Ct,2> gx, Array<$Ct,2> logx) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= logx.length()) return;

        int ndIdx[2];
        logx.idx2ndIdx(ndIdx, idx);
        int i = ndIdx[0];
        int j = ndIdx[1];
        if (p[j] > 0) {
            $Ct delta = (i == p[j]-1) ? 1 : 0;
            gx(i,j) += gy[j] * (exp(logx(i,j)) - delta);
        }
    }""")
    quote
        gdims, bdims = cudims(length(logx))
        $k(gdims, bdims, pointer(gy), pointer(p), gx, logx)
    end
end

@generated function ∇softmax_crossentropy!(gy::CuVector{T}, p::CuMatrix{T}, gx::CuMatrix{T}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void softmax_crossentropy_grad($Ct *gy, $Ct *p, $Ct *gx, Array<$Ct,2> logx) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= logx.length()) return;

        int ndIdx[2];
        logx.idx2ndIdx(ndIdx, idx);
        int i = ndIdx[0];
        int j = ndIdx[1];
        gx[idx] += gy[j] * (exp(logx[idx]) - p[idx]);
    }""")
    quote
        gdims, bdims = cudims(length(logx))
        $k(gdims, bdims, pointer(gy), pointer(p), pointer(gx), logx)
    end
end
