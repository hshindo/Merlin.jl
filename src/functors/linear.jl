export Linear

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
f = Linear(Float32, 10, 7)
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
  LinearFun(zerograd(w), zerograd(b))
end

@compat (f::LinearFun)(x::Var) = Linear()(f.w, f.b, x)


type Linear; end

@compat(f::Linear)(w::Var, b::Var, x::Var) = forward(f, (w,b,x))

@compat function (f::Linear){T}(w::Matrix{T}, b::Vector{T}, x::Matrix{T})
  y = w * x
  broadcast!(.+, y, y, b)
  f, y
end

@compat function (f::Linear){T}(w::CuMatrix{T}, b::CuVector{T}, x::CuMatrix{T})
  throw("Not implemented yet.")
end

function backward!{T}(f::Linear, vw::Var{Matrix{T}}, vb::Var{Vector{T}}, vx::Var{Matrix{T}})
  y, gy = data(vy)
  b, gb = data(vb)
  x, gx = data(vx)
  if isdefined(vw, :grad)
    w, gw = data(vw)
    BLAS.gemm!('N', 'T', T(1), gy, x, T(1), gw)
  end
  isempty(gx) || BLAS.gemm!('T', 'N', T(1), w, gy, T(1), gx)
  for offset = 1:length(b):length(gy)
    BLAS.axpy!(length(b), T(1), pointer(gy,offset), stride(gy,1), pointer(gb), stride(gb,1))
  end
end
