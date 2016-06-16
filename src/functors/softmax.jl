export softmax, logsoftmax

type Softmax; end
type LogSoftmax; end

doc"""
    softmax(x)

Compute softmax along the second axis.
Currently, only 2-d is supported.

$ p(x) = {\exp(f(x)) \over \sum_{x_2} \exp(f(x))} $
"""
softmax(x::Var) = forward(Softmax(), x)

"""
    logsoftmax(x)

Compute logarithm of softmax along the second axis.
Currently, only 2-d is supported.
"""
logsoftmax(x::Var) = forward(LogSoftmax(), x)

@compat (f::Softmax){T}(x::Matrix{T}) = f, softmax(x)

@compat (f::LogSoftmax){T}(x::Matrix{T}) = f, logsoftmax(x)

@compat function (f::Softmax)(x::CuArray)
  CUDNN.softmax!(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, x, similar(x))
end
@compat function (f::LogSoftmax)(x::CuArray)
  CUDNN.softmax!(CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, x, similar(x))
end

function backward!{T}(f::Softmax, x, gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
  isempty(gx) && return
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

function backward!(f::Softmax, x, gx::CuArray, y, gy)
  isempty(gx) && return
  CUDNN.∇softmax!(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, y, gy, gx; beta=1.0)
end

function backward!{T}(f::LogSoftmax, x, gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
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

function ∇softmax2!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
  # d yi / d xj = yi * (delta (i=j) - yj)
  g = y .* gy
  sumdx = sum(g, 1)
  g -= y .* sumdx
  copy!(gx, g)
end

# experimental JIT compile
function softmax_native{T}(x::Matrix{T})
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
