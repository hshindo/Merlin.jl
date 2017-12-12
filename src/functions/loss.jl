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
    y = isvoid(p.data,q:data) ? nothing : crossentropy(p.data,q.data)
    Var(y, (crossentropy,p,q))
end

function crossentropy(p::Vector{Int}, q::Matrix{T}) where T
    y = Array{T}(length(p))
    @inbounds for i = 1:length(p)
        y[i] = p[i] > 0 ? -log(q[p[i],i]) : T(0)
    end
    y
end

function addgrad!(y::Var, ::typeof(crossentropy), p::Var, q::Var)
    isvoid(q.grad) || ∇crossentropy!(y.grad, p.data, q.data, q.grad)
end

function ∇crossentropy!(gy::Vector{T}, p::Vector{Int}, q::Matrix{T}, gq::Matrix{T}) where T
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
    if isvoid(x.data)
        y = nothing
    else
        T = eltype(x)
        y = [mapreduce(x -> x*x, +, x.data) * T(lambda) / 2]
    end
    Var(y, (l2,x,lambda))
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
    y = isvoid(x1.data,x2.data) ? nothing : mse(x1.data,x2.data)
    Var(y, (mse,x1,x2))
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
    isvoid(p.data,x.data) && return Var(nothing,(softmax_crossentropy,p,x))
    y, logx = softmax_crossentropy(p.data, x.data)
    Var(y, (softmax_crossentropy,p,x,logx))
end

function softmax_crossentropy(p::Vector{Int}, x::Matrix{T}) where T
    length(p) == size(x,2) || throw("Length unmatch.")
    logx = logsoftmax(x)
    y = Array{T}(length(p))
    @inbounds for i = 1:length(p)
        y[i] = p[i] > 0 ? -logx[p[i],i] : T(0)
    end
    y, logx
end

function softmax_crossentropy(p::Matrix{T}, x::Matrix{T}) where T
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

function ∇softmax_crossentropy!(gy::Vector{T}, p::Vector{Int}, logq::Matrix{T}, gq::Matrix{T}) where T
    @inbounds for j = 1:length(p)
        p[j] > 0 || continue
        for i = 1:size(logq,1)
            delta = i == p[j] ? T(1) : T(0)
            gq[i,j] += gy[j] * (exp(logq[i,j]) - delta)
        end
    end
end

function ∇softmax_crossentropy!(gy::Vector{T}, p::Matrix{T}, logx::Matrix{T}, gx::Matrix{T}) where T
    @inbounds for j = 1:size(p,2)
        for i = 1:size(logx,1)
            gx[i,j] += gy[j] * (exp(logx[i,j]) - p[i,j])
        end
    end
end

function ∇softmax_crossentropy!(gy::Matrix{T}, p::Matrix{Int}, q::Matrix{T}, gq::Matrix{T}) where T
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
