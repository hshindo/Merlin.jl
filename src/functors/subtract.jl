type Subtract <: Functor
end
type ElemSubtract <: Functor
end

import Base.-
-(arg1::Variable, arg2::Variable) = Subtract()(arg1, arg2)
-(arg1::Variable, arg2::Any) = Subtract()(arg1, arg2)
-(arg1::Any, arg2::Variable) = Subtract()(arg1, arg2)
-(arg::Variable) = 0 - arg

import Base.(.-)
.-(arg1::Variable, arg2::Variable) = ElemSubtract()(arg1, arg2)
-(arg1::Variable, arg2::Any) = ElemSubtract()(arg1, arg2)
-(arg1::Any, arg2::Variable) = ElemSubtract()(arg1, arg2)

@compat (f::Subtract)(arg1, arg2) = forward(f, arg1, arg2)
function forward!(f::Subtract, v::Variable)
  v.value = v[1].value - v[2].value
  v.backward! = () -> begin
    T = eltype(v)
    hasgrad(v[1]) && axpy!(T(1), v.grad, v[1].grad)
    hasgrad(v[2]) && axpy!(T(-1), v.grad, v[2].grad)
  end
end

@compat (f::ElemSubtract)(arg1, arg2) = forward(f, arg1, arg2)
function forward!(f::ElemSubtract, v::Variable)
  v.value = v[1].value .- v[2].value
  v.backward! = () -> begin
    T = eltype(v)
    #hasgrad(v[1]) && axpy!(T(1), v.grad, v[1].grad)
    #hasgrad(v[2]) && axpy!(T(-1), v.grad, v[2].grad)
  end
end
