export softmax

const SOFTMAX_F32 = Libdl.dlsym(libmerlin, :softmax_f32)
const ∇SOFTMAX_F32 = Libdl.dlsym(libmerlin, :softmax_f32_grad)

softmax_handle(::Type{Float32}) = SOFTMAX_F32
∇softmax_handle(::Type{Float32}) = ∇SOFTMAX_F32

"""
    softmax(x::Var)

Returns a softmax over the `ndims(x)-1`-th dimension.

```math
f(x) = \exp(x) \over \sum \exp(x)
```
"""
function softmax(x::Var{<:Array})
    data = softmax(x.data)
    Var(data, x.batchdims, softmax, (x,))
end
softmax(x::Node) = Node(softmax, x)

function addgrad!(y::Var{<:Array}, ::typeof(softmax), x::Var)
    isvoid(x.grad) && return
    ∇softmax!(y.data, y.grad, x.grad)
end

function softmax{T}(x::Array{T})
    y = similar(x)
    h = softmax_handle(T)
    dims = size3d(x, ndims(x)-1)
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint), x, y, dims[1], dims[2], dims[3])
    y
end

function softmax_jl{T}(x::Matrix{T})
    y = similar(x)
    for j = 1:size(x,2)
        maxv = x[1,j]
        @inbounds for i = 1:size(x,1)
            maxv = max(maxv, x[i,j])
        end

        z = T(0)
        @inbounds for i = 1:size(x,1)
            y[i,j] = exp(x[i,j] - maxv)
            z += y[i,j]
        end
        z == T(0) && error("z == 0")
        invz = 1 / z
        @inbounds for i = 1:size(x,1)
            y[i,j] *= invz
        end
    end
    y
end

function ∇softmax!{T}(y::Array{T}, gy::Array{T}, gx::Array{T})
    h = ∇softmax_handle(T)
    dims = size3d(y, ndims(y)-1)
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{T},Cint,Cint,Cint), y, gy, gx, dims[1], dims[2], dims[3])
end

function ∇softmax_jl!{T}(y::Matrix{T}, gy::Matrix{T}, gx::Matrix{T})
    for j = 1:size(y,2)
        sum = T(0)
        for i = 1:size(y,1)
            sum += gy[i,j] * y[i,j]
        end
        for i = 1:size(y,1)
            gx[i,j] += y[i,j] * (gy[i,j]-sum)
        end
    end
end
