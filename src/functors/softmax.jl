export softmax

"""
Computes softmax along the second axis.

```math
f(x) = \frac{\exp(x_{i})}{\sum_{j}^{n}\exp(x_{j})}
```

```math
p(x) = {\\exp(f(x)) \\over \\sum_{x_2} \\exp(f(x))}
```

## 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = softmax(x)
```
"""
type Softmax <: Functor
end

function forward(f::Softmax, args::Vector{Var})
  x = args[1]
  y = softmax(x.val)
  backward! = gy -> hasgrad(x) && ∇softmax!(x.grad, y, gy)
  Var(y, nothing, f, args, backward!)
end

"""
Computes softmax
"""
softmax(x::Var) = Softmax()(x)

function softmax{T}(x::Matrix{T})
  CT = "float"
  size1, size2 = size(x)
  sym = Symbol(join(["softmax",CT,size1,size2], "_"))
  if isdefined(Native, sym)
    h = eval(Native, sym)
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
    h = Native.compile(src, sym)
  end
  y = similar(x)
  ccall(h, Void, (Ptr{T},Ptr{T}), x, y)
  y
end

function softmax2{T}(x::Matrix{T})
  y = similar(x)
  max = maximum(x, 1)
  for j = 1:size(x,2)
    z = T(0)
    @inbounds @simd for i = 1:size(x,1)
      z += exp(x[i,j] - max[j])
    end
    z == T(0) && error("z == 0")
    @inbounds @simd for i = 1:size(x,1)
      y[i,j] = exp(x[i,j] - max[j]) / z
    end
  end
  y
end

function softmax(x::CuArray)
  y = similar(x)
  CUDNN.softmax!(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, x, y)
end

function ∇softmax2!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
  # d yi / d xj = yi * (delta (i=j) - yj)
  g = y .* gy
  sumdx = sum(g, 1)
  g -= y .* sumdx
  copy!(gx, g)
end

function ∇softmax!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
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
  CUDNN.∇softmax!(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, y, gy, gx; beta=1.0)
end
