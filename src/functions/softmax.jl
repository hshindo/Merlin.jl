export softmax, logsoftmax

matrix(x::UniArray) = reshape(x, size(x,1), prod(Base.tail(size(x))))
matrix(x::UniMatrix) = x

doc"""
    softmax(x, dim::Int)

Softmax function over the given dimension.

```math
f(x) = \exp(x) \over \sum \exp(x)
```
"""
function softmax(x::Var; dim::Int)
    ydata = softmax(x.data, dim)
    Var(ydata, ∇softmax!, (x,dim))
end

function softmax(x::Var, dims::Vector{Int})
    @assert ndims(x) == 2
    h = pack(x, dims, typemin(eltype(x)))
    h = softmax(h, dim=ndims(x))
    unpack(h, dims)
end

function softmax(x::Matrix, dim::Int)
    if dim == 1
        softmax(x)
    elseif dim == 2
        tx = Array(transpose(x))
        ty = softmax(tx)
        Array(transpose(ty))
    end
end

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

function softmax(x::CuArray, dim::Int)
    if dim == ndims(x)-1
        mode = CUDNN.CUDNN_SOFTMAX_MODE_CHANNEL
    elseif dim == ndims(x)
        mode = CUDNN.CUDNN_SOFTMAX_MODE_INSTANCE
    end
    CUDNN.softmax(x, mode=mode)
end

function ∇softmax!(y::Var, x::Var, dim::Int)
    isnothing(x.grad) && return
    ∇softmax!(y.data, y.grad, x.grad)
end

function ∇softmax!(y::CuArray, gy::CuArray, gx::CuArray)
    CUDNN.∇softmax!(y, gy, gx)
end

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
    ydata = logsoftmax(matrix(x.data))
    ydata = reshape(ydata, size(x))
    Var(ydata, ∇logsoftmax!, (x,))
end

function logsoftmax(x::CuArray{T}) where T
    CUDNN.softmax(x, algo=CUDNN.CUDNN_SOFTMAX_LOG)
end

function logsoftmax(x::Matrix{T}) where T
    y = similar(x)
    max = maximum(x, dims=1)
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

function ∇logsoftmax!(y::Var, x::Var)
    isnothing(x.grad) && return
    ∇logsoftmax!(matrix(y.data), matrix(y.grad), matrix(x.grad))
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
