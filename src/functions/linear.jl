export Linear

doc"""
    Linear(w::Var, b::Var)

Create an object of linear function (a.k.a. affine transformation).

$ f(x) = w * x + b $

## Arguments
* `w::Var`: weight matrix
* `b::Var`: bias vector
"""
type Linear
  w::Var
  b::Var
end

"""
    Linear(::Type{T}, indim::Int, outdim::Int)

## Arguments
* `T`: element type
* `indim`: input dimension
* `outdim`: output dimension

## 👉 Example
```julia
x = Var(rand(Float32,10,5))
f = Linear(Float32, 10, 7)
y = f(x)
```
"""
function Linear{T}(::Type{T}, indim::Int, outdim::Int)
  x = sqrt(6 / (indim + outdim))
  r = rand(outdim, indim) * 2x - x
  w = convert(Matrix{T}, r)
  b = fill(T(0), outdim, 1)
  Linear(param(w), param(b))
end

@compat (f::Linear)(x::Var) = linear(f.w, x, f.b)

function linear(w::Var, x::Var, b::Var)
  @checkargs linear (w,x,b)
  y = linear(w.value, x.value, b.value)
  df(gy) = begin
    hasgrad(w) && BLAS.gemm!('N', 'T', T(1), gy, x, T(1), gw)
    hasgrad(x) && BLAS.gemm!('T', 'N', T(1), w, gy, T(1), gx)
  end
  Var(y, df, [w,x,b])
end

function linear{T}(w::Matrix{T}, x::Matrix{T}, b::Matrix{T})
  y = w * x
  #broadcast!(.+, y, y, b)
  y
end

function linear{T}(w::CuMatrix{T}, x::CuMatrix{T}, b::CuMatrix{T})
  y = w * x
  #CUDNN.add!(b, y)
  y
end

function ∇linear!{T}(w::Matrix{T}, gw::Matrix{T}, b, gb,
  x::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})

  hasgrad(w) && BLAS.gemm!('N', 'T', T(1), gy, x, T(1), gw)
  hasgrad(x) && BLAS.gemm!('T', 'N', T(1), w, gy, T(1), gx)
  for offset = 1:length(b):length(gy)
    BLAS.axpy!(length(b), T(1), pointer(gy,offset), stride(gy,1), pointer(gb), stride(gb,1))
  end
end
