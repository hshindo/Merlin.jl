export Subtract
export ElemSubtract
using Base.LinAlg.BLAS # necessary?

type Subtract <: Functor
end
type ElemSubtract <: Functor
end

import Base.-
-(v1::Variable, v2::Variable) = (v1, v2) |> Subtract()
-(a::Number, v::Variable) = (Variable(a,nothing), v) |> Subtract()
-(v::Variable, a::Number) = (v, Variable(a,nothing)) |> Subtract()
-(x::Data, v::Variable) = (x, v) |> Subtract()
-(v::Variable, x::Data) = (v, x) |> Subtract()
-(v::Variable) = 0 - v

import Base.(.-)
.-(v1::Variable, v2::Variable) = (v1, v2) |> ElemSubtract()
.-(a::Number, v::Variable) = (Variable(a,nothing), v) |> ElemSubtract()
.-(v::Variable, a::Number) = (v, Variable(a,nothing)) |> ElemSubtract()
.-(x::Data, v::Variable) = (x, v) |> ElemSubtract()
.-(v::Variable, x::Data) = (v, x) |> ElemSubtract()

@compat (f::Subtract)(args) = forward(f, args)
function forward!(f::Subtract, v::Variable)
  v.value = v[1].value - v[2].value
  v.backward! = () -> begin
    T = eltype(v)
    hasgrad(v[1]) && axpy!(T(1), v.grad, v[1].grad)
    hasgrad(v[2]) && axpy!(T(-1), v.grad, v[2].grad)
  end
end

@compat (f::ElemSubtract)(args) = forward(f, args)
function forward!(f::ElemSubtract, v::Variable)
  v.value = v[1].value .- v[2].value
  v.backward! = () -> begin
    T = eltype(v)
    hasgrad(v[1]) && backward!(f, 1.0, v[1].grad, v.grad)
    hasgrad(v[2]) && backward!(f, -1.0, v[2].grad, v.grad)
  end
end

function backward!{T,N}(f::ElemSubtract, a::Float64, gx::Array{T,N}, gy::Array{T,N})
  for offset = 1:length(gx):length(gy)
    axpy!(length(gx), T(a), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
  end
end
