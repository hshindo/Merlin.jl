export linear, Linear

type Linear
  w::Var
  b::Var
end

@compat function (f::Linear)(args::Vector{Var})
  w, b, x = f.w, f.b, args[1]
  y = w.value * x.value
  broadcast!(.+, y, b.value)
  function df(gy)
    T = eltype(gy)
    hasgrad(w) && BLAS.gemm!('N', 'T', T(1), gy, x.grad, T(1), w.grad)
    hasgrad(x) && BLAS.gemm!('T', 'N', T(1), w.value, gy, T(1), x.grad)
    #s = sum(gy, 2)
    #broadcast!(+, gb, s)
  end
  Var(y, df, [w,x,b])
end

function Linear{T}(::Type{T}, indim::Int, outdim::Int)
  x = sqrt(6 / (indim + outdim))
  r = rand(outdim, indim) * 2x - x
  w = convert(Matrix{T}, r)
  b = fill(T(0), outdim, 1)
  Linear(Var(w), Var(b))
end

doc"""
    linear(w, x, [b])

Compute linear function (a.k.a. affine transformation).

$ f(x) = w * x + b $
where $w$, $x$, $b$ are matrices.

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
f = Linear(Float32,10,7)
y = f(x)
```
"""
linear(w, x, b=Var()) = forward(Linear(), [w,x,b])

#=
function ∇linear!(w::Var, b::Var, x::Var)
  w, gw, b, gb = w.value, w.grad, b.value, b.grad
  hasgrad(v[1]) && BLAS.gemm!('T', 'N', T(1), w, gy, T(1), v[1].grad)
  BLAS.gemm!('N', 'T', T(1), gy, v[1].value, T(1), gw)
  for offset = 1:length(b):length(gy)
    BLAS.axpy!(length(b), T(1), pointer(gy,offset), stride(gy,1), pointer(gb), stride(gb,1))
  end
end
=#

mat(a::Array) = reshape(a, size(a, 1), length(a)÷size(a,1))
isvec(a::Array) = ndims(a) == 2 && size(a, 2) == 1
