type Add <: Functor
end
type ElemAdd <: Functor
end

import Base.+
+(v1::Variable, v2::Variable) = Add()([v1,v2])
import Base.(.+)
.+(v1::Variable, v2::Variable) = ElemAdd()([v1,v2])

function compile(f::Add, var::Variable)
end

@compat (f::Add)(args) = forward(f, args)
function forward!(f::Add, v::Variable)
  @assert (length(v.args) == 2)
  v.value = v[1].value + v[2].value
  v.backward! = () -> begin
    backward!(f, v[1].grad, v.grad)
    backward!(f, v[2].grad, v.grad)
  end
end

backward!{T,N}(f::Add, gx::Array{T,N}, gy::Array{T,N}) = axpy!(T(1), gy, gx)
backward!{T,N}(f::Add, gx::Void, gy::Array{T,N}) = ()

@compat (f::ElemAdd)(args) = forward(f, args)
function forward!(f::ElemAdd, v::Variable)
  @assert (length(v.args) == 2)
  v.value = v[1].value .+ v[2].value
  v.backward! = () -> begin
    backward!(f, v[1].grad, v.grad)
    backward!(f, v[2].grad, v.grad)
  end
end

function backward!{T,N}(f::ElemAdd, gx::Array{T,N}, gy::Array{T,N})
  if length(gx) == length(gy)
    axpy!(T(1), gy, gx)
  else
    for offset = 1:length(gx):length(gy)
      axpy!(length(gx), T(1), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
    end
  end
end
backward!{T,N}(f::ElemAdd, gx::Void, gy::Array{T,N}) = ()
