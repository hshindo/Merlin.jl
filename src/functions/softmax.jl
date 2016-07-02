export softmax

#const SOFTMAX_FW_F32 = Libdl.dlsym(library, :softmax_fw_f32)
#const SOFTMAX_FW_F64 = Libdl.dlsym(library, :softmax_fw_f64)
#const SOFTMAX_BW_F32 = Libdl.dlsym(library, :softmax_bw_f32)
#const SOFTMAX_BW_F64 = Libdl.dlsym(library, :softmax_bw_f64)
const SOFTMAX_FW_F32 = Libdl.dlsym(libmerlin, :softmax_float)
const SOFTMAX_FW_F64 = Libdl.dlsym(libmerlin, :softmax_double)

#softmax_handle(::Type{Float32}) = SOFTMAX_FW_F32, SOFTMAX_BW_F32
#softmax_handle(::Type{Float64}) = SOFTMAX_FW_F64, SOFTMAX_BW_F64
softmax_handle(::Type{Float32}) = SOFTMAX_FW_F32, SOFTMAX_FW_F32
softmax_handle(::Type{Float64}) = SOFTMAX_FW_F64, SOFTMAX_FW_F64

doc"""
    softmax(x::Var, dim::Int)

Compute softmax along the given axis.

$ p(x) = {\exp(f(x)) \over \sum_{x_2} \exp(f(x))} $
"""
softmax(x::Var, dim::Int) = Softmax(dim)(x)

type Softmax
  dim::Int
end

@compat function (f::Softmax)(x::Var)
  @checkargs f (x,)
  y = softmax(x.value)
  df(gy) = hasgrad(x) && ∇softmax!(x.grad, y, gy)
  Var(y, df, [x])
end

function softmax{T}(x::Array{T}, dim::Int)
  @assert 0 < dim <= ndims(x)
  h = softmax_handle(T)[1]
  y = similar(x)
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint}), x, y, splitdims(x,dim))
  y
end

function softmax_jl{T}(x::Matrix{T})
  y = similar(x)
  #max = maximum(x, 1)
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

function softmax(x::CuArray)
  softmax!(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, x, similar(x))
end

function ∇softmax!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
  h = softmax_handle(T)[2]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{T},Cint,Cint), gx, y, gy, size(gx,1), size(gx,2))
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

function ∇softmax!(gx::CuArray, y::CuArray, gy::CuArray)
  ∇softmax!(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, y, gy, gx; beta=1.0)
end
