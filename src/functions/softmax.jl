export softmax

const SOFTMAX_FW_F32 = Libdl.dlsym(library, :softmax_fw_f32)
const SOFTMAX_FW_F64 = Libdl.dlsym(library, :softmax_fw_f64)
const SOFTMAX_BW_F32 = Libdl.dlsym(library, :softmax_bw_f32)
const SOFTMAX_BW_F64 = Libdl.dlsym(library, :softmax_bw_f64)

softmax_handle(::Type{Float32}) = SOFTMAX_FW_F32, SOFTMAX_BW_F32
softmax_handle(::Type{Float64}) = SOFTMAX_FW_F64, SOFTMAX_BW_F64

doc"""
    softmax(x::Var, dim::Int)

Compute softmax along the second axis.
Currently, only 2-d is supported.

$ p(x) = {\exp(f(x)) \over \sum_{x_2} \exp(f(x))} $
"""
softmax(x::Var, dim::Int) = Softmax(dim)(x)

type Softmax
  dim::Int
end

@compat function (f::Softmax)(x::Var)
  @checkargs f (x,)
  @assert f.dim == 1
  y = softmax(x.value)
  df(gy) = hasgrad(x) && ∇softmax!(x.grad, y, gy)
  Var(y, df, [x])
end

function softmax{T}(x::Matrix{T})
  h = softmax_handle(T)[1]
  y = similar(x)
  ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint), x, y, size(x,1), size(x,2))
  y
end

function softmax_jl{T}(x::Matrix{T})
  y = similar(x)
  #max = maximum(x, 1)
  for j = 1:size(x,2)
    maxv = x[1,j]
    @inbounds @simd for i = 1:size(x,1)
      x[i,j] > maxv && (maxv = x[i,j])
    end

    z = T(0)
    @inbounds @simd for i = 1:size(x,1)
      y[i,j] = exp_approx(x[i,j] - maxv)
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

# https://github.com/pluskid/Mocha.jl/blob/be17557e2db3a81d2ca517d0dc4a0488b4935285/src/layers/softmax.jl
function softmax_mocha{T}(input::Array{T}, dim::Int, output::Array{T})
  dim_pre, dim_prob, dim_post = splitdims(input, dim)
  for i = 0:dim_pre-1
    for j = 0:dim_post-1
      idx = Int[i + dim_pre*(k + dim_prob*j) for k=0:dim_prob-1] + 1

      maxval = -Inf
      for k in idx
        @inbounds maxval = max(maxval, input[k])
      end
      for k in idx
        @inbounds output[k] = exp(input[k]-maxval)
      end
      the_sum = 0.0
      for k in idx
        @inbounds the_sum += output[k]
      end
      for k in idx
        @inbounds output[k] /= the_sum
      end
    end
  end
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
