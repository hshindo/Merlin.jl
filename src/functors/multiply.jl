export Multiply

type Multiply <: Functor
end

function Base.call(f::Multiply, arg1::Variable, arg2::Variable)
  y = arg1.value * arg2.value
  getgrad = gy -> ∇multiply(arg1.value, arg2.value, gy)
  Variable(f, [arg1,arg2], y, getgrad)
end

import Base.*
*(v1::Variable, v2::Variable) = Multiply()(v1, v2)

function ∇multiply{T}(x1::Matrix{T}, x2::Matrix{T}, gy::Matrix{T})
  gx1 = gemm('N', 'T', gy, x2)
  gx2 = gemm('T', 'N', x1, gy)
  [gx1, gx2]
end
