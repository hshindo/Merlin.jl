export linear, Linear

doc"""
    Linear(w::Var, b::Var)

Compute linear function (a.k.a. affine transformation or fully-connected layer).

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
type Linear
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
