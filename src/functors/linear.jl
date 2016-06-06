export linear, Linear

type Linear
  w::Var
  b::Var
end

function Linear{T}(::Type{T}, indim::Int, outdim::Int)
  x = sqrt(6 / (indim + outdim))
  r = rand(outdim, indim) * 2x - x
  w = convert(Matrix{T}, r)
  b = fill(T(0), outdim, 1)
  Linear(param(w), param(b))
end

@compat function (f::Linear)(args::Vector{Var})
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
  Var(y, df, [w,b,x])
end
@compat (f::Linear)(x::Var) = forward(f, [f.w,f.b,x])

doc"""
    linear(w, x, b)

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
linear(w::Var, x::Var, b::Var) = Linear(w,b)(x)

mat(a::Array) = reshape(a, size(a, 1), length(a)÷size(a,1))
isvec(a::Array) = ndims(a) == 2 && size(a, 2) == 1
