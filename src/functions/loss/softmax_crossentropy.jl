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
x = Var(rand(Float32,10,5))
y = softmax_crossentropy(p, x)
```
"""
function softmax_crossentropy(x::Var, y::Var)
    configure!(x, y)
    logx = logsoftmax(x.data)
    l = softmax_crossentropy(logx, y.data)
    Var(l, (softmax_crossentropy,x,y,logx))
end

function softmax_crossentropy(logx::Matrix{T}, y::Vector{Int}) where T
    size(logx,2) == length(y) || throw("Length unmatch.")
    l = Array{T}(length(y))
    @inbounds for i = 1:length(y)
        l[i] = y[i] > 0 ? -logx[y[i],i] : T(0)
    end
    l
end

function softmax_crossentropy(logx::Matrix{T}, y::Matrix{T}) where T
    size(logx) == size(y) || throw("Size mismatch.")
    l = Array{T}(size(y,2))
    @inbounds for j = 1:size(y,2)
        s = T(0)
        for i = 1:size(y,1)
            s += -y[i,j] * logx[i,j]
        end
        l[j] = s
    end
    l
end

function addgrad!(l::Var, ::typeof(softmax_crossentropy), x::Var, y::Var, logx)
    isvoid(x.grad) && return
    ∇softmax_crossentropy!(l.grad, x.grad, y.data, logx)
end

function ∇softmax_crossentropy!(gl::Vector{T}, gx::Matrix{T}, y::Vector{Int}, logx::Matrix{T}) where T
    @inbounds for j = 1:length(y)
        y[j] <= 0 && continue
        for i = 1:size(logx,1)
            delta = i == y[j] ? T(1) : T(0)
            gx[i,j] += gl[j] * (exp(logx[i,j]) - delta)
        end
    end
end

function ∇softmax_crossentropy!(gl::Vector{T}, gx::Matrix{T}, y::Matrix{T}, logx::Matrix{T}) where T
    @inbounds for j = 1:size(y,2)
        for i = 1:size(logx,1)
            gx[i,j] += gl[j] * (exp(logx[i,j]) - y[i,j])
        end
    end
end

@generated function softmax_crossentropy(logx::CuMatrix{T}, y::CuVector{Cint}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void softmax_crossentropy($Ct *l, Array<$Ct,2> logx, int *y, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            l[idx] = y[idx] > 0 ? -logx(y[idx]-1,idx) : 0;
        }
    }""")
    quote
        size(logx,2) == length(y) || throw("Length unmatch.")
        l = CuArray{T}(length(y))
        gdims, bdims = cudims(length(l))
        $k(gdims, bdims, rawpointer(l), logx, rawpointer(y), length(l))
        l
    end
end

@generated function softmax_crossentropy(logx::CuMatrix{T}, y::CuMatrix{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void softmax_crossentropy($Ct *l, $Ct *logx, $Ct *y, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            l[idx] = -y[idx] * logx[idx];
        }
    }""")
    quote
        size(logx) == size(y) || throw("Length unmatch.")
        l = similar(y)
        gdims, bdims = cudims(length(l))
        $k(gdims, bdims, rawpointer(l), rawpointer(logx), rawpointer(y), length(l))
        vec(sum(l,1))
    end
end

@generated function ∇softmax_crossentropy!(gl::CuVector{T}, gx::CuMatrix{T}, y::CuVector{Cint}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void softmax_crossentropy_grad($Ct *gl, Array<$Ct,2> gx, int *y, Array<$Ct,2> logx) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= logx.length()) return;

        int ndIdx[2];
        logx.ind2sub(ndIdx, idx);
        int i = ndIdx[0];
        int j = ndIdx[1];
        if (y[j] > 0) {
            $Ct delta = (i == y[j]-1) ? 1 : 0;
            gx(i,j) += gl[j] * (exp(logx(i,j)) - delta);
        }
    }""")
    quote
        gdims, bdims = cudims(length(logx))
        $k(gdims, bdims, rawpointer(gl), gx, rawpointer(y), logx)
    end
end

@generated function ∇softmax_crossentropy!(gl::CuVector{T}, gx::CuMatrix{T}, y::CuMatrix{T}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void softmax_crossentropy_grad($Ct *gl, $Ct *gx, $Ct *y, Array<$Ct,2> logx) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= logx.length()) return;

        int ndIdx[2];
        logx.ind2sub(ndIdx, idx);
        int i = ndIdx[0];
        int j = ndIdx[1];
        gx[idx] += gl[j] * (exp(logx[idx]) - y[idx]);
    }""")
    quote
        gdims, bdims = cudims(length(logx))
        $k(gdims, bdims, rawpointer(gl), rawpointer(gx), rawpointer(y), logx)
    end
end
