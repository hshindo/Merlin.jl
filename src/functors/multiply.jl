type Multiply <: Functor
end
type ElemMultiply <: Functor
end

import Base.*
*(arg1::Variable, arg2::Variable) = Multiply()(arg1, arg2)
*(arg1::Any, arg2::Variable) = Multiply()(arg1, arg2)
*(arg1::Variable, arg2::Any) = Multiply()(arg2, arg1)

import Base.(.*)
.*(arg1::Variable, arg2::Variable) = ElemMultiply()(arg1, arg2)
.*(arg1::Any, arg2::Variable) = ElemMultiply()(arg1, arg2)
.*(arg1::Variable, arg2::Any) = ElemMultiply()(arg2, arg1)

@compat (f::Multiply)(arg1, arg2) = forward(f, arg1, arg2)
function forward!(f::Multiply, v::Variable)
  v.value = v[1].value * v[2].value
  v.backward! = () -> begin
    T = eltype(v)
    hasgrad(v[1]) && gemm!('N', 'T', T(1), v.grad, v[2].value, T(1), v[1].grad)
    hasgrad(v[1]) && gemm!('T', 'N', T(1), v[1].value, v.grad, T(1), v[2].grad)
  end
end

@compat (f::ElemMultiply)(arg1, arg2) = forward(f, arg1, arg2)
function forward!(f::ElemMultiply, v::Variable)
  v.value = v[1].value .* v[2].value
  v.backward! = () -> begin
    T = eltype(v)
    hasgrad(v[1]) && backward!(f, v[2].value. v[1].grad, v.grad)
    hasgrad(v[2]) && backward!(f, v[1].value. v[2].grad, v.grad)
  end
end

function backward!{T,N}(f::ElemMultiply, x2::Array{T,N}, gx1::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(gx)
    gx1[i] += gy[i] * x2[i]
  end
end
