type Multiply <: Functor
end
type ElemMultiply <: Functor
end
type Multiply! <: Functor
  a::Float64
end

import Base.*
function *(v1::Variable, v2::Variable)
  if typeof(v1.f) == Add!
    Multiply!()
    Add!(1.)(v1, v2)
  elseif typeof(v2.f) == Add!
    Add!(1.)(v2, v1)
  else
    Add!(1.)(Variable(0,nothing), v1) + v2
  end
  Multiply()([v1,v2])
end
*(x::Any, v::Variable) = Variable(x,nothing) * v
*(v::Variable, x::Any) = v * Variable(x,nothing)

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

@compat (f::Multiply!)(arg1, arg2) = forward(f, arg1, arg2)
function forward!(f::Multiply!, v::Variable)
  @assert (length(v.args) == 3)
  T = eltype(v)
  v.value = gemm!('N', 'N', T(1), v[2].value, v[3].value, T(1), v[1].value)
  v.backward! = () -> begin
    #T = eltype(v)
    #v[1].grad == nothing || gemm!('N', 'T', T(1), v.grad, v[2].value, T(1), v[1].grad)
    #v[2].grad == nothing || gemm!('T', 'N', T(1), v[1].value, v.grad, T(1), v[2].grad)
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
