export softmax

const SOFTMAX_F32 = Libdl.dlsym(libmerlin, :softmax_float)
const SOFTMAX_F64 = Libdl.dlsym(libmerlin, :softmax_double)
const ∇SOFTMAX_F32 = Libdl.dlsym(libmerlin, :softmax_grad_float)
const ∇SOFTMAX_F64 = Libdl.dlsym(libmerlin, :softmax_grad_double)

softmax_handle(::Type{Float32}) = SOFTMAX_F32, ∇SOFTMAX_F32
softmax_handle(::Type{Float64}) = SOFTMAX_F64, ∇SOFTMAX_F64

"""
    softmax(x::Var, dim::Int)
"""
function softmax(x::Var, dim::Int)
    y = softmax(x.data, dim)
    df(gy) = ∇softmax!(x.grad, y, gy, dim)
    Var(y, [x], softmax, df)
end

softmax(x::GraphNode, dim::Int) = GraphNode(softmax, x, dim)

function softmax{T}(x::Array{T}, dim::Int)
    @assert 0 < dim <= ndims(x)
    h = softmax_handle(T)[1]
    y = similar(x)
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint}), x, y, splitdims(x,dim))
    y
end

function softmax(x::CuArray)
    softmax!(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, x, similar(x))
end

function ∇softmax!{T}(gx::Array{T}, y::Array{T}, gy::Array{T}, dim::Int)
    h = softmax_handle(T)[2]
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{T},Ptr{Cint}), gx, y, gy, splitdims(gx,dim))
    y
end

function ∇softmax!(gx::CuArray, y::CuArray, gy::CuArray)
    ∇softmax!(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, y, gy, gx; beta=1.0)
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
