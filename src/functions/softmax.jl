export softmax, logsoftmax

const SOFTMAX_F32 = Libdl.dlsym(libmerlin, :softmax_f32)
const ∇SOFTMAX_F32 = Libdl.dlsym(libmerlin, :softmax_f32_grad)

const LOGSOFTMAX_F32 = Libdl.dlsym(libmerlin, :logsoftmax_f32)
const ∇LOGSOFTMAX_F32 = Libdl.dlsym(libmerlin, :logsoftmax_f32_grad)

softmax_handle(::Type{Float32}) = SOFTMAX_F32
∇softmax_handle(::Type{Float32}) = ∇SOFTMAX_F32

logsoftmax_handle(::Type{Float32}) = LOGSOFTMAX_F32
∇logsoftmax_handle(::Type{Float32}) = ∇LOGSOFTMAX_F32

"""
    softmax(x::Var)

Returns a softmax over the `ndims(x)-1`-th dimension.

```math
f(x) = \exp(x) \over \sum \exp(x)
```
"""
function softmax(x::Var)
    y = Var(nothing, softmax, (x,))
    softmax!(y, x.data)
    y
end

function softmax!(out::Var, x::Array)
    out.data = softmax(x)
    out.df! = function df!()
        isvoid(out[1].grad) || ∇softmax!(out.data, out.grad, out[1].grad)
    end
end

"""
    logsoftmax(x::Var)

Returns a logarithm of softmax function.
"""
function logsoftmax(x::Var)
    y = Var(nothing, logsoftmax, (x,))
    logsoftmax!(y, x.data)
    y
end


function logsoftmax!(out::Var, x::Array)
    out.data = logsoftmax(x)
    out.df! = function df!()
        isvoid(out[1].grad) || ∇logsoftmax!(out.data, out.grad, out[1].grad)
    end
end

function softmax{T}(x::Array{T})
    y = similar(x)
    h = softmax_handle(T)
    dims = dim3d(x, ndims(x)-1)
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint), x, y, dims[1], dims[2], dims[3])
    y
end

function ∇softmax!{T}(y::Array{T}, gy::Array{T}, gx::Array{T})
    h = ∇softmax_handle(T)
    dims = dim3d(y, ndims(y)-1)
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{T},Cint,Cint,Cint), y, gy, gx, dims[1], dims[2], dims[3])
end

function logsoftmax{T}(x::Array{T})
    y = similar(x)
    h = logsoftmax_handle(T)
    dims = dim3d(x, ndims(x)-1)
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint), x, y, dims[1], dims[2], dims[3])
    y
end

function ∇logsoftmax!{T}(y::Array{T}, gy::Array{T}, gx::Array{T})
    h = ∇logsoftmax_handle(T)
    dims = dim3d(y, ndims(y)-1)
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{T},Cint,Cint,Cint), y, gy, gx, dims[1], dims[2], dims[3])
end

function dim3d(x::Array, dim::Int)
    dim1, dim2, dim3 = 1, size(x,dim), 1
    for i = 1:dim-1
        dim1 *= size(x, i)
    end
    for i = dim+1:ndims(x)
        dim3 *= size(x, i)
    end
    (dim1, dim2, dim3)
end

function softmax_jl{T}(x::Matrix{T})
    y = similar(x)
    for j = 1:size(x,2)
        maxv = x[1,j]
        @inbounds @simd for i = 1:size(x,1)
            maxv = max(maxv, x[i,j])
        end

        z = T(0)
        @inbounds @simd for i = 1:size(x,1)
            y[i,j] = exp(x[i,j] - maxv)
            z += y[i,j]
        end
        z == T(0) && error("z == 0")
        invz = 1 / z
        @inbounds @simd for i = 1:size(x,1)
            y[i,j] *= invz
        end
    end
    y
end

function ∇softmax_jl!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
    # d yj / d xi = yj * (delta (i=j) - yi)
    for d = 1:size(gx,2)
        for i = 1:size(gx,1)
            yi = y[i,d]
            for j = 1:size(gx,1)
                delta = i == j ? T(1) : T(0)
                gx[i,d] += gy[j,d] * y[j,d] * (delta - yi)
            end
        end
    end
end

function logsoftmax_jl{T}(x::Matrix{T})
    y = similar(x)
    max = maximum(x, 1)
    for j = 1:size(x,2)
        sum = T(0)
        @inbounds @simd for i = 1:size(x,1)
            sum += exp(x[i,j] - max[j])
        end
        logz = log(sum)
        @inbounds @simd for i = 1:size(x,1)
            y[i,j] = x[i,j] - max[j] - logz
        end
    end
    y
end
