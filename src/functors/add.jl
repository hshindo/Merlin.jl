export Add

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
function forward!{T,N}(f::Add, v::Variable)
  v.value = mapreduce(a -> a.value, +, v.args)
  v.backward! = () -> begin
    for a in v.args
      axpy!(T(1), v.grad, a.grad)
    end
  end
end

import Base.+
+(v1::Variable, v2::Variable) = Add()([v1,v2])
