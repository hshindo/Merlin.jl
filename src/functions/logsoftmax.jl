export logsoftmax

const LOGSOFTMAX_F32 = Libdl.dlsym(libmerlin, :logsoftmax_float)
#const LOGSOFTMAX_FW_F64 = Libdl.dlsym(library, :logsoftmax_fw_f64)
#const LOGSOFTMAX_BW_F32 = Libdl.dlsym(library, :logsoftmax_bw_f32)
#const LOGSOFTMAX_BW_F64 = Libdl.dlsym(library, :logsoftmax_bw_f64)

logsoftmax_handle(::Type{Float32}) = LOGSOFTMAX_F32, LOGSOFTMAX_F32
#logsoftmax_handle(::Type{Float64}) = LOGSOFTMAX_FW_F64, LOGSOFTMAX_BW_F64

"""
    logsoftmax(x::Var, dim::Int)

Compute logarithm of softmax along the given axis.
"""
logsoftmax(x::Var, dim::Int) = LogSoftmax(dim)(x)

type LogSoftmax
  dim::Int
end

@compat function (f::LogSoftmax)(x::Var)
  @checkargs f (x,)
  y = logsoftmax(x.value, f.dim)
  df(gy) = hasgrad(x) && ∇logsoftmax!(x.grad, y, gy, f.dim)
  Var(y, df, [x])
end

function logsoftmax{T}(x::Array{T}, dim::Int)
  @assert 0 < dim <= ndims(x)
  h = logsoftmax_handle(T)[1]
  y = similar(x)
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint}), x, y, splitdims(x,dim))
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

function logsoftmax(x::CuArray)
  softmax!(CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, x, similar(x))
end

function ∇logsoftmax!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
  h = logsoftmax_handle(T)[2]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{T},Cint,Cint), gx, y, gy, size(gx,1), size(gx,2))
  y
end
