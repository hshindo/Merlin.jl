export softmax, logsoftmax

const SOFTMAX_FW_F32 = Libdl.dlsym(library, :softmax_fw_f32)
const SOFTMAX_FW_F64 = Libdl.dlsym(library, :softmax_fw_f64)
const SOFTMAX_BW_F32 = Libdl.dlsym(library, :softmax_bw_f32)
const SOFTMAX_BW_F64 = Libdl.dlsym(library, :softmax_bw_f64)
const LOGSOFTMAX_FW_F32 = Libdl.dlsym(library, :logsoftmax_fw_f32)
const LOGSOFTMAX_FW_F64 = Libdl.dlsym(library, :logsoftmax_fw_f64)
const LOGSOFTMAX_BW_F32 = Libdl.dlsym(library, :logsoftmax_bw_f32)
const LOGSOFTMAX_BW_F64 = Libdl.dlsym(library, :logsoftmax_bw_f64)

softmax_handle(::Type{Float32}) = SOFTMAX_FW_F32, SOFTMAX_BW_F32
softmax_handle(::Type{Float64}) = SOFTMAX_FW_F64, SOFTMAX_BW_F64
logsoftmax_handle(::Type{Float32}) = LOGSOFTMAX_FW_F32, LOGSOFTMAX_BW_F32
logsoftmax_handle(::Type{Float64}) = LOGSOFTMAX_FW_F64, LOGSOFTMAX_BW_F64

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

function softmax2{T}(x::Matrix{T})
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

function softmax(x::CuArray)
  softmax!(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, x, similar(x))
end

function ∇softmax!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
  h = softmax_handle(T)[2]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{T},Cint,Cint), gx, y, gy, size(gx,1), size(gx,2))
  y
end

function ∇softmax2!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
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

function ∇softmax2!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
  # d yi / d xj = yi * (delta (i=j) - yj)
  g = y .* gy
  sumdx = sum(g, 1)
  g -= y .* sumdx
  broadcast!(+, gx, gx, g)
end

function ∇softmax!(gx::CuArray, y::CuArray, gy::CuArray)
  ∇softmax!(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, y, gy, gx; beta=1.0)
end


"""
    logsoftmax(x)

Compute logarithm of softmax along the second axis.
Currently, only 2-d is supported.
"""
function logsoftmax(x::Var)
  @checkargs logsoftmax (x,)
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

function logsoftmax2{T}(x::Matrix{T})
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

function ∇logsoftmax2!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
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

# experimental JIT compile
function softmax_native_jit{T}(x::Matrix{T})
  CT = "float"
  size1, size2 = size(x)
  sym = Symbol(join(["softmax",CT,size1,size2], "_"))
  if isdefined(Merlin, sym)
    h = eval(sym)
  else
    src = """
    #include <algorithm>
    #include <math.h>
    using namespace std;

    float Exp(float x) { return expf(x); }
    double Exp(double x) { return exp(x); }
    float Log(float x) { return logf(x); }
    double Log(double x) { return log(x); }

    extern "C" {
      void run($CT *x, $CT *y) {
        for (int m2 = 0; m2 < $(size2); m2++) {
          int offset = m2*$(size1);
          $CT x_max = x[offset];
          for (int m1 = 1; m1 < $(size1); m1++) x_max = std::max(x_max, x[m1 + offset]);

            $CT sum = static_cast<$CT>(0);
          for (int m1 = 0; m1 < $(size1); m1++) {
            int i = m1 + offset;
            y[i] = Exp(x[i] - x_max);
            sum += y[i];
          }

          $CT invsum = 1 / sum;
          for (int m1 = 0; m1 < $(size1); m1++) y[m1 + offset] *= invsum;
        }
      }
    }"""
    h = cppcompile(src, sym)
  end
  y = similar(x)
  ccall(h, Void, (Ptr{T},Ptr{T}), x, y)
  y
end
