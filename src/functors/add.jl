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

#=
function Base.call(f::Add, args::Vector{Variable})
  xs = map(a -> a.value, args)
  y = reduce(+, args)
  getgrad = gy -> map(_ -> gy, xs)
  Variable(f, args, y, getgrad)
end
Base.call(f::Add, args...) = call(f, [args...])
=#

function forward{T,N}(f::Add, xs::Vector{Array{T,N}})
  y = reduce(+, xs)
  backward = gy -> map(_ -> gy, xs)
  y, backward
end

import Base.+
+(v1::Variable, v2::Variable) = Add()([v1,v2])
