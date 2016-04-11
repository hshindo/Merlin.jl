export Add
export Subtract
export Multiply

type Add <: Functor
end

function compile(f::Add, var::Variable)
  args = Variable[]
  for a in var.args
    if typeof(a.f) == Add
      append!(args, a.args)
    else
      push!(args, a)
    end
  end
  Variable(nothing, Add(), args)
end

@compat (f::Add)(args) = forward(f, args)
function forward!(f::Add, v::Variable)
  v.value = mapreduce(a -> a.value, +, v.args)
  v.backward! = () -> begin
    T = eltype(v.value)
    for a in v.args
      axpy!(T(1), v.grad, a.grad)
    end
  end
end

import Base.+
+(v1::Variable, v2::Variable) = Add()([v1,v2])


type Subtract <: Functor
end

@compat (f::Subtract)(args) = forward(f, args)
function forward!(f::Subtract, v::Variable)
  @assert (length(v.args) == 2)
  v.value = v[1].value - v[2].value
  v.backward! = () -> begin
    T = eltype(v.value)
    axpy!(T(1), v.grad, v[1].grad)
    axpy!(T(-1), v.grad, v[2].grad)
  end
end

import Base.-
-(v1::Variable, v2::Variable) = Subtract()([v1,v2])


type Multiply <: Functor
end

@compat (f::Multiply)(args) = forward(f, args)
function forward!{T,N}(f::Multiply, v::Variable)
  @assert (length(v.args) == 2)
  v.value = v[1].value * v[2].value
  v.backward! = () -> begin
    gemm!('N', 'T', T(1), v.grad, v[2].value, T(1), v[1].grad)
    gemm!('T', 'N', T(1), v[1].value, v.grad, T(1), v[2].grad)
  end
end

import Base.*
*(v1::Variable, v2::Variable) = Multiply()([v1,v2])
