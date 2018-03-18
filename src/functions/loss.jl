export crossentropy, l2, mse, softmax_crossentropy

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
    Var(crossentropy(p.data,q.data), (crossentropy,p,q))
end

function crossentropy(p::Vector{I}, q::Matrix{T}) where {I<:Integer,T}
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
    isvoid(q.grad) || ∇crossentropy!(y.grad, p.data, q.data, q.grad)
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

doc"""
    l2(x::Var, lambda::Float64)

L2 regularization.

```math
y = \frac{\lambda}{2}\left\Vert \mathbf{x} \right\Vert ^{2}
```

```julia
x = Var(rand(Float32,10,5))
y = l2(x, 0.01)
```
"""
function l2(x::Var, lambda::Float64)
    T = eltype(x)
    y = mapreduce(x -> x*x, +, x.data) * T(lambda) / 2
    Var([y], (l2,x,lambda))
end

function addgrad!(y::Var, ::typeof(l2), x::Var, lambda::Float64)
    T = eltype(y)
    isvoid(y.grad) || ∇l2!(y.grad, x.data, x.grad, T(lambda))
end

function ∇l2!(gy::Vector{T}, x::Array{T}, gx::Array{T}, lambda::T) where T
    @inbounds for i = 1:length(x)
        gx[i] += gy[1] * lambda * x[i]
    end
end

doc"""
    mse(x1, x2)

Mean Squared Error function between `x1` and `x2`.
The mean is calculated over the minibatch.
Note that the error is not scaled by 1/2.
"""
function mse(x1::Var, x2::Var)
    Var(mse(x1.data,x2.data), (mse,x1,x2))
end

function mse(x1::Matrix{T}, x2::Matrix{T}) where T
    size(x1) == size(x2) || throw("Size unmatch.")
    y = similar(x1, size(x1,2))
    for j = 1:size(x1,2)
        v = T(0)
        for i = 1:size(x1,1)
            d = x1[i,j] - x2[i,j]
            v += d * d
        end
        y[j] = v / size(x1,1)
    end
    y
end

function addgrad!(y::Var, ::typeof(mse), x1::Var, x2::Var)
    T = eltype(y)
    gx1 = isvoid(x1.grad) ? Array{T}(0,0) : x1.grad
    gx2 = isvoid(x2.grad) ? Array{T}(0,0) : x2.grad
    ∇mse!(y.grad, x1.data, gx1, x2.data, gx2)
end

function ∇mse!(gy::Vector{T}, x1::Matrix{T}, gx1::Matrix{T}, x2::Matrix{T}, gx2::Matrix{T}) where T
    for j = 1:size(x1,2)
        for i = 1:size(x1,1)
            g = gy[j] * (x1[i,j]-x2[i,j]) * 2 / size(x1,1)
            isempty(gx1) || (gx1[i,j] += g)
            isempty(gx2) || (gx2[i,j] -= g)
        end
    end
end

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
    @assert isvoid(p.grad)
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

@generated function softmax_crossentropy(p::CuVector{Cint}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)

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
        culaunch($f, gdims, bdims, Ptr{T}(y), Ptr{Cint}(p), logx, Cint(length(y)))
        y
    end
end

@generated function softmax_crossentropy(p::CuMatrix{T}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)

    __global__ void f($Ct *y, $Ct *p, $Ct *logx, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            y[idx] = -p[idx] * logx[idx];
        }
    }""")
    quote
        size(p) == size(logx) || throw("Length unmatch.")
        y = similar(p)
        gdims, bdims = cudims(length(y))
        culaunch($f, gdims, bdims, Ptr{T}(y), Ptr{T}(p), Ptr{T}(logx), length(y))
        vec(sum(y,1))
    end
end

function addgrad!(y::Var, ::typeof(softmax_crossentropy), p::Var, x::Var, work)
    isvoid(x.grad) || ∇softmax_crossentropy!(y.grad, p.data, x.grad, work)
end

function ∇softmax_crossentropy!(gy::Vector{T}, p::Vector{I}, gx::Matrix{T}, logx::Matrix{T}) where {T,I<:Integer}
    @inbounds for j = 1:length(p)
        p[j] > 0 || continue
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

@generated function ∇softmax_crossentropy!(gy::CuVector{T}, p::CuVector{Cint}, gx::CuMatrix{T}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)

    __global__ void f($Ct *gy, int *p, Array<$Ct,2> gx, Array<$Ct,2> logx) {
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
        culaunch($f, gdims, bdims, Ptr{T}(gy), Ptr{Cint}(p), gx, logx)
    end
end

@generated function ∇softmax_crossentropy!(gy::CuVector{T}, p::CuMatrix{T}, gx::CuMatrix{T}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)

    __global__ void f($Ct *gy, $Ct *p, $Ct *gx, Array<$Ct,2> logx) {
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
        culaunch($f, gdims, bdims, Ptr{T}(gy), Ptr{Cint}(p), Ptr{T}(gx), logx)
    end
end

function ∇softmax_crossentropy2!(gy::Matrix{T}, p::Matrix{Int32}, q::Matrix{T}, gq::Matrix{T}) where T
    @inbounds for i = 1:length(p)
        p[i] > 0 || continue
        if q[p[i],i] < T(-1e-10) || q[p[i],i] > T(1e-10)
            gq[p[i],i] -= T(1) / q[p[i],i]
        end
    end
end
