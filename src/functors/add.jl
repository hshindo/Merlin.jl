export Add
export ElemAdd

type Add <: Functor
end
type ElemAdd <: Functor
end

import Base.+
+(v1::Variable, v2::Variable) = (v1, v2) |> Add()
+(a::Number, v::Variable) = (Variable(a,nothing), v) |> Add()
+(v::Variable, a::Number) = a + v
+(x::Data, v::Variable) = (x, v) |> Add()
+(v::Variable, x::Data) = (x, v) |> Add()

@compat (f::Add)(args) = forward(f, args)
function forward!(f::Add, v::Variable)
  v.value = v[1].value + v[2].value
  v.backward! = () -> begin
    T = eltype(v)
    hasgrad(v[1]) && axpy!(T(1), v.grad, v[1].grad)
    hasgrad(v[2]) && axpy!(T(1), v.grad, v[2].grad)
  end
end

@compat (f::ElemAdd)(arg1, arg2) = forward(f, arg1, arg2)
function forward!(f::ElemAdd, v::Variable)
  v.value = v[1].value .+ v[2].value
  v.backward! = () -> begin
    hasgrad(v[1]) && backward!(f, v[1].grad, v.grad)
    hasgrad(v[2]) && backward!(f, v[2].grad, v.grad)
  end
end

function backward!{T,N}(f::ElemAdd, gx::Array{T,N}, gy::Array{T,N})
  for offset = 1:length(gx):length(gy)
    axpy!(length(gx), T(1), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
  end
end
