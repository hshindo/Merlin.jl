type Add <: Functor
end
type ElemAdd <: Functor
end
type Add! <: Functor
  a::Float64
end

import Base.+
function +(v1::Variable, v2::Variable)
  if typeof(v1.f) == Add!
    Add!(1.)(v1, v2)
  elseif typeof(v2.f) == Add!
    Add!(1.)(v2, v1)
  else
    Add!(1.)(Variable(0,nothing), v1) + v2
  end
end
+(v::Variable, x::Any) = v + Variable(x,nothing)
+(x::Any, v::Variable) = Variable(x,nothing) + v

import Base.-
#-(v1::Variable, v2::Variable)
#-(x::Any, v::Variable) = Add!(-1.)([Variable(x),v])
#-(v::Variable, x::Any) = Add!(-1.)([v,Variable(x)])

import Base.(.+)
.+(v1::Variable, v2::Variable) = ElemAdd()([v1,v2])
.+(x::Any, v::Variable) = ElemAdd()([Variable(x),v])
.+(v::Variable, x::Any) = ElemAdd()([v,Variable(x)])

function topdown(var::Variable)

end

function compile(f::Add, var::Variable)
end

@compat (f::Add)(arg1, arg2) = forward(f, [arg1,arg2])
function forward!(f::Add, v::Variable)
  @assert (length(v.args) == 2)
  v.value = v[1].value + v[2].value
  v.backward! = () -> begin
    hasgrad(v[1]) && backward!(f, v[1].grad, v.grad)
    hasgrad(v[2]) && backward!(f, v[2].grad, v.grad)
  end
end
function forward!{T,N}(f::Add, y::Array{T,N})

end

backward!{T,N}(f::Add, gx::Array{T,N}, gy::Array{T,N}) = axpy!(T(1), gy, gx)

@compat (f::Add!)(arg1, arg2) = forward(f, arg1, arg2)
function forward!(f::Add!, v::Variable)
  @assert (length(v.args) == 2)
  v.value = add!(f.a, v[1].value, v[2].value)
  v.backward! = () -> begin
    #hasgrad(v[1]) && backward!(f, v[1].grad, v.grad)
    #hasgrad(v[2]) && backward!(f, v[2].grad, v.grad)
  end
end

add!(a::Float64, x1::Number, x2::Array) = x1 + x2
add!(a::Float64, x1::Array, x2::Number) = x1 + x2
add!(a::Float64, x1::Array, x2::Array) = axpy!(a, x2, x1)

@compat (f::ElemAdd)(args) = forward(f, args)
function forward!(f::ElemAdd, v::Variable)
  @assert (length(v.args) == 2)
  v.value = v[1].value .+ v[2].value
  v.backward! = () -> begin
    hasgrad(v[1]) && backward!(f, v[1].grad, v.grad)
    hasgrad(v[2]) && backward!(f, v[2].grad, v.grad)
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
