export logsoftmax

const LOGSOFTMAX_F32 = Libdl.dlsym(libmerlin, :logsoftmax_float)
const LOGSOFTMAX_F64 = Libdl.dlsym(libmerlin, :logsoftmax_double)
const ∇LOGSOFTMAX_F32 = Libdl.dlsym(libmerlin, :logsoftmax_grad_float)
const ∇LOGSOFTMAX_F64 = Libdl.dlsym(libmerlin, :logsoftmax_grad_double)

logsoftmax_handle(::Type{Float32}) = LOGSOFTMAX_F32, ∇LOGSOFTMAX_F32
logsoftmax_handle(::Type{Float64}) = LOGSOFTMAX_F64, ∇LOGSOFTMAX_F64

"""
    logsoftmax(x::Var, dim::Int)
"""
@graph function logsoftmax(x::Var, dim::Int)
    y = logsoftmax(x.data, dim)
    df(gy) = isconst(x) || ∇logsoftmax!(x.grad, y, gy, dim)
    Var(y, [x], df)
end

function logsoftmax{T}(x::Array{T}, dim::Int)
    @assert 0 < dim <= ndims(x)
    h = logsoftmax_handle(T)[1]
    y = similar(x)
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint}), x, y, splitdims(x,dim))
    y
end

function logsoftmax(x::CuArray)
    softmax!(CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, x, similar(x))
end

function ∇logsoftmax!{T}(gx::Array{T}, y::Array{T}, gy::Array{T}, dim::Int)
    h = logsoftmax_handle(T)[2]
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{T},Ptr{Cint}), gx, y, gy, splitdims(gx,dim))
    y
end

function logsoftmax_jl{T}(x::Matrix{T}, dim::Int)
    @assert dim == 1
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
