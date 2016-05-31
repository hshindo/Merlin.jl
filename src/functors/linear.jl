export Linear

"""
Compute linear transformation.

### 👉 Example
```julia
f = Linear(Float32, 10, 3)
x = Var(rand(Float32,10,5))
y = f(x)
```
"""
type Linear
  w::Var
  b::Var
end

@compat (f::Linear)(x::Var) = f.w * x + f.b

function Linear{T}(::Type{T}, indim::Int, outdim::Int)
  x = sqrt(6 / (indim + outdim))
  r = rand(outdim, indim) * 2x - x
  w = convert(Matrix{T}, r)
  b = fill(T(0), outdim, 1)
  Linear(Var(w,grad=true), Var(b,grad=true))
end

mat(a::Array) = reshape(a, size(a, 1), length(a)÷size(a,1))
isvec(a::Array) = ndims(a) == 2 && size(a, 2) == 1

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

#=
function forward!(f::Linear, v::Variable)
  v.value = f.w.value * v[1].value .+ f.b.value
  push!(v.args, f.w)
  push!(v.args, f.b)
  v.backward! = () -> begin
    # dy / dx = w^T * gy, dy / dw = gy * x^T
    T = eltype(v.value)
    gy = v.grad
    w, gw, b, gb = f.w.value, f.w.grad, f.b.value, f.b.grad
    hasgrad(v[1]) && BLAS.gemm!('T', 'N', T(1), w, gy, T(1), v[1].grad)
    BLAS.gemm!('N', 'T', T(1), gy, v[1].value, T(1), gw)
    for offset = 1:length(b):length(gy)
      BLAS.axpy!(length(b), T(1), pointer(gy,offset), stride(gy,1), pointer(gb), stride(gb,1))
    end
  end
end
=#
