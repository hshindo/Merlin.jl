export logsoftmax

const LOGSOFTMAX_FW_F32 = Libdl.dlsym(library, :logsoftmax_fw_f32)
const LOGSOFTMAX_FW_F64 = Libdl.dlsym(library, :logsoftmax_fw_f64)
const LOGSOFTMAX_BW_F32 = Libdl.dlsym(library, :logsoftmax_bw_f32)
const LOGSOFTMAX_BW_F64 = Libdl.dlsym(library, :logsoftmax_bw_f64)

logsoftmax_handle(::Type{Float32}) = LOGSOFTMAX_FW_F32, LOGSOFTMAX_BW_F32
logsoftmax_handle(::Type{Float64}) = LOGSOFTMAX_FW_F64, LOGSOFTMAX_BW_F64

"""
    logsoftmax(x::Var, dim::Int)

Compute logarithm of softmax along the second axis.
Currently, only 2-d is supported.
"""
logsoftmax(x::Var, dim::Int) = LogSoftmax(dim)(x)

type LogSoftmax
  dim::Int
end

@compat function (f::LogSoftmax)(x::Var)
  @checkargs f (x,)
  @assert f.dim == 2
  y = logsoftmax(x.value)
  df(gy) = hasgrad(x) && ∇logsoftmax!(x.grad, y, gy)
  Var(y, df, [x])
end

function logsoftmax{T}(x::Matrix{T})
  h = logsoftmax_handle(T)[1]
  y = similar(x)
  ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint), x, y, size(x,1), size(x,2))
  y
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

function logsoftmax(x::CuArray)
  softmax!(CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, x, similar(x))
end

function ∇logsoftmax!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
  h = logsoftmax_handle(T)[2]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{T},Cint,Cint), gx, y, gy, size(gx,1), size(gx,2))
  y
end

function ∇logsoftmax_jl!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
  # d yj / d xi = delta(i=j) - exp(yi)
  for d = 1:size(gx,2)
    for i = 1:size(gx,1)
      expy = exp(y[i,d])
      for j = 1:size(gx,1)
        delta = i == j ? T(1) : T(0)
        gx[i,d] += gy[j,d] * (delta - expy)
      end
    end
  end
end
