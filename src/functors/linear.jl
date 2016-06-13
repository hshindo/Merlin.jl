export Linear

doc"""
    Linear(w::Var, b::Var)

Linear function (a.k.a. affine transformation).

$ f(x) = w * x + b $

## Arguments
* `w::Var`: weight matrix
* `b::Var`: bias vector

## 👉 Example
```julia
x = Var(rand(Float32,10,5))
f = Linear(Float32, 10, 7)
y = f(x)
```
"""
type Linear <: Functor
  w::Var
  b::Var
end

"""
    Linear(::Type{T}, indim::Int, outdim::Int)
"""
function Linear{T}(::Type{T}, indim::Int, outdim::Int)
  x = sqrt(6 / (indim + outdim))
  r = rand(outdim, indim) * 2x - x
  w = convert(Matrix{T}, r)
  b = fill(T(0), outdim, 1)
  Linear(param(w), param(b))
end

function forward{T<:Array}(f::Linear, xs::Vector{T})
  w, b, x = xs[1], xs[2], xs[3]
  y = w * x
  broadcast!(.+, y, y, b)
  f, y
end

function backward!{T}(f::Linear, xs, gxs, y, gy::Array{T})
  w, b, x = xs[1], xs[2], xs[3]
  gw, gb, gx = gxs[1], gxs[2], gxs[3]
  isempty(gw) || BLAS.gemm!('N', 'T', T(1), gy, x, T(1), gw)
  isempty(gx) || BLAS.gemm!('T', 'N', T(1), w, gy, T(1), gx)
  for offset = 1:length(b):length(gy)
    BLAS.axpy!(length(b), T(1), pointer(gy,offset), stride(gy,1), pointer(gb), stride(gb,1))
  end
end

@compat (f::Linear)(x::Var) = forward(f, [f.w,f.b,x])

#=
@compat function (f::Linear)(args::Vector{Var})
  @checkargs f args
  w, b, x = args[1], args[2], args[3]
  y = w.value * x.value
  broadcast!(.+, y, y, b.value)
  function df(gy)
    T = eltype(gy)
    hasgrad(w) && BLAS.gemm!('N', 'T', T(1), gy, x.value, T(1), w.grad)
    hasgrad(x) && BLAS.gemm!('T', 'N', T(1), w.value, gy, T(1), x.grad)
    for offset = 1:length(b.value):length(gy)
      BLAS.axpy!(length(b.value), T(1), pointer(gy,offset), stride(gy,1), pointer(b.grad), stride(b.grad,1))
    end
  end
  Var(y, df, args)
end
@compat (f::Linear)(x::Var) = f([f.w,f.b,x])
=#
