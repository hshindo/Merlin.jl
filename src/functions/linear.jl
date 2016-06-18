export LinearFun, linear

doc"""
    LinearFun(w::Var, b::Var)

Linear function (a.k.a. affine transformation).

$ f(x) = w * x + b $

## Arguments
* `w::Var`: weight matrix
* `b::Var`: bias vector

## 👉 Example
```julia
x = Var(rand(Float32,10,5))
f = LinearFun(Float32, 10, 7)
y = f(x)
```
"""
type LinearFun
  w::Var
  b::Var
end

function LinearFun{T}(::Type{T}, indim::Int, outdim::Int)
  x = sqrt(6 / (indim + outdim))
  r = rand(outdim, indim) * 2x - x
  w = convert(Matrix{T}, r)
  b = fill(T(0), outdim, 1)
  LinearFun(param(w), param(b))
end

@compat (f::LinearFun)(x::Var) = linear(f.w, x, f.b)

function linear(w::Var, x::Var, b::Var)
  @checkargs linear (w,x,b)
  y = linear(w.value, x.value, b.value)
  df(gy) = begin
    T = eltype(gy)
    hasgrad(w) && BLAS.gemm!('N', 'T', T(1), gy, x.value, T(1), w.grad)
    hasgrad(x) && BLAS.gemm!('T', 'N', T(1), w.value, gy, T(1), x.grad)
    #for offset = 1:length(b):length(gy)
    #  BLAS.axpy!(length(b), T(1), pointer(gy,offset), stride(gy,1), pointer(gb), stride(gb,1))
    #end
  end
  Var(y, df, [w,x,b])
end

function linear{T}(w::Matrix{T}, x::Matrix{T}, b::Matrix{T})
  y = w * x
  broadcast!(.+, y, y, b)
  y
end

function linear{T}(w::CuMatrix{T}, x::CuMatrix{T}, b::CuMatrix{T})
  throw("Not implemented yet.")
end
