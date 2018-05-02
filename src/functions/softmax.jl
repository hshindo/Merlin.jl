export softmax, logsoftmax

"""
    softmax(x, dim::Int)

Softmax function over the given dimension.

```math
f(x) = \exp(x) \over \sum \exp(x)
```
"""
function softmax(x::Var)
    configure!(x)
    Var(softmax(x.data), (softmax,x))
end
softmax(x::CuArray) = CUDNN.softmax(x)

function softmax(x::Matrix{T}) where T
    y = similar(x)
    @inbounds for j = 1:size(x,2)
        maxv = x[1,j]
        for i = 1:size(x,1)
            maxv = max(maxv, x[i,j])
        end
        z = T(0)
        for i = 1:size(x,1)
            y[i,j] = exp(x[i,j] - maxv)
            z += y[i,j]
        end
        z == T(0) && throw("z == 0")
        invz = 1 / z
        for i = 1:size(x,1)
            y[i,j] *= invz
        end
    end
    y
end

function addgrad!(y::Var, ::typeof(softmax), x::Var)
    isvoid(x.grad) && return
    ∇softmax!(y.data, y.grad, x.grad)
end

∇softmax!(y::CuArray, gy::CuArray, gx::CuArray) = CUDNN.∇softmax!(y, gy, gx)

function ∇softmax!(y::Matrix{T}, gy::Matrix{T}, gx::Matrix{T}) where T
    @inbounds for j = 1:size(y,2)
        sum = T(0)
        for i = 1:size(y,1)
            sum += gy[i,j] * y[i,j]
        end
        for i = 1:size(y,1)
            gx[i,j] += y[i,j] * (gy[i,j]-sum)
        end
    end
end

"""
    logsoftmax(x)

Logarithm of softmax function.
"""
function logsoftmax(x::Var)
    configure!(x)
    Var(logsoftmax(x.data), (logsoftmax,x))
end
logsoftmax(x::CuArray) = CUDNN.softmax(x, CUDNN.CUDNN_SOFTMAX_LOG)

function logsoftmax(x::Matrix{T}) where T
    y = similar(x)
    max = maximum(x, 1)
    @inbounds for j = 1:size(x,2)
        sum = T(1e-10)
        for i = 1:size(x,1)
            sum += exp(x[i,j] - max[j])
        end
        logz = log(sum)
        for i = 1:size(x,1)
            y[i,j] = x[i,j] - max[j] - logz
        end
    end
    y
end

function addgrad!(y::Var, ::typeof(logsoftmax), x::Var)
    isvoid(x.grad) && return
    ∇logsoftmax!(y.data, y.grad, x.grad)
end

function ∇logsoftmax!(y::CuArray, gy::CuArray, gx::CuArray)
    CUDNN.∇softmax!(y, gy, gx, CUDNN.CUDNN_SOFTMAX_LOG)
end

function ∇logsoftmax!(y::Matrix{T}, gy::Matrix{T}, gx::Matrix{T}) where T
    @inbounds for j = 1:size(y,2)
        sum = T(0)
        for i = 1:size(y,1)
            sum += gy[i,j]
        end
        for i = 1:size(y,1)
            gx[i,j] += gy[i,j] - exp(y[i,j]) * sum
        end
    end
end
