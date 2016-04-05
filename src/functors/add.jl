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
  Variable(Add(), args, nothing)
end

function Base.call(f::Add, args::Vector{Variable})
  x1, x2 = args[1].value, args[2].value
  y = broadcast(+, x1, x2)
  backward! =
  Variable(f, args, y, backward!)
end
Base.call(f:Add, args...) = call(f, [args...])

import Base.+
+(v1::Variable, v2::Variable) = Add()(v1, v2)

function ∇add!{T}(gx::Array{T}, gy::Array{T})
  if ndims(gx) > ndims(gy) && length(gx) > length(gy)
    error("")
    broadcast!(+, gx, gx, gy)
  else
    for offset = 1:length(gx):length(gy)
      axpy!(length(gx), T(1.0), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
    end
  end
end
