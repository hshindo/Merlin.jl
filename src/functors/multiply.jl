type Multiply <: Functor
end
type ElemMultiply <: Functor
end

import Base.*
*(v1::Variable, v2::Variable) = Multiply()([v1,v2])
import Base.(.*)
.*(v1::Variable, v2::Variable) = ElemMultiply()([v1,v2])

@compat (f::Multiply)(args) = forward(f, args)
function forward!(f::Multiply, v::Variable)
  @assert (length(v.args) == 2)
  v.value = v[1].value * v[2].value
  v.backward! = () -> begin
    T = eltype(v)
    v[1].grad == nothing || gemm!('N', 'T', T(1), v.grad, v[2].value, T(1), v[1].grad)
    v[2].grad == nothing || gemm!('T', 'N', T(1), v[1].value, v.grad, T(1), v[2].grad)
  end
end

@compat (f::ElemMultiply)(args) = forward(f, args)
function forward!(f::ElemMultiply, v::Variable)
  @assert (length(v.args) == 2)
  v.value = v[1].value .* v[2].value
  v.backward! = () -> begin
    v[1].grad == nothing || backward!(f, v[2].value. v[1].grad, v.grad)
    v[2].grad == nothing || backward!(f, v[1].value. v[2].grad, v.grad)
  end
end

function backward!{T,N}(f::ElemMultiply, x2::Array{T,N}, gx1::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(v.grad)
    gx1[i] += gy[i] * x2[i]
  end
end
