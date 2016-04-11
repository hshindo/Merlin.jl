export Multiply

type Multiply <: Functor
end

@compat (f::Multiply)(args) = forward(f, args)
function forward!{T,N}(f::Multiply, v::Variable)
  v.value = v[1].value * v[2].value
  v.backward! = () -> begin
    T = eltype(v.value)
    gemm!('N', 'T', T(1), v.grad, v[2].value, T(1), v[1].grad)
    gemm!('T', 'N', T(1), v[1].value, v.grad, T(1), v[2].grad)
  end
end

import Base.*
*(v1::Variable, v2::Variable) = Multiply()([v1,v2])
