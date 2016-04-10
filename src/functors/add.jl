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

function forward{T,N}(f::Add, xs::Vector{Array{T,N}})
  y = reduce(+, xs)
  backward! = (gxs, gy) -> begin
    for gx in gxs
      axpy!(T(1), gy, gx)
    end
  end
  y, backward!
end

import Base.+
+(v1::Variable, v2::Variable) = Add()([v1,v2])
