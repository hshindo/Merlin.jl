type Subtract <: Functor
end
type ElemSubtract <: Functor
end

import Base.-
-(v1::Variable, v2::Variable) = Subtract()([v1,v2])
import Base.(.-)
.-(v1::Variable, v2::Variable) = ElemSubtract()([v1,v2])

@compat (f::Subtract)(args) = forward(f, args)
function forward!(f::Subtract, v::Variable)
  @assert (length(v.args) == 2)
  v.value = v[1].value - v[2].value
  v.backward! = () -> begin
    backward!(f, 1.0, v[1].grad, v.grad)
    backward!(f, -1.0, v[2].grad, v.grad)
  end
end

backward!{T,N}(f::Subtract, a, gx::Array{T,N}, gy::Array{T,N}) = axpy!(T(a), gy, gx)
backward!{T,N}(f::Subtract, a, gx::Void, gy::Array{T,N}) = ()

@compat (f::ElemSubtract)(args) = forward(f, args)
function forward!(f::ElemSubtract, v::Variable)
  @assert (length(v.args) == 2)
  v.value = v[1].value .- v[2].value
  v.backward! = () -> begin
    backward!(f, 1.0, v[1].grad, v.grad)
    backward!(f, -1.0, v[2].grad, v.grad)
  end
end

function backward!{T,N}(f::ElemSubtract, a, gx::Array{T,N}, gy::Array{T,N})
  if length(gx) == length(gy)
    axpy!(T(a), gy, gx)
  else
    for offset = 1:length(gx):length(gy)
      axpy!(length(gx), T(a), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
    end
  end
end
backward!{T,N}(f::ElemSubtract, a, gx::Void, gy::Array{T,N}) = ()
