export Multiply
export ElemMultiply
using Base.LinAlg.BLAS # necessary?

type Multiply <: Functor
end
type ElemMultiply <: Functor
end

import Base.*
*(v1::Variable, v2::Variable) = (v1, v2) |> Multiply()
*(a::Number, v::Variable) = (Variable(a,nothing), v) |> Multiply()
*(v::Variable, a::Number) = (v, Variable(a,nothing)) |> Multiply()
*(x::Data, v::Variable) = (x, v) |> Multiply()
*(v::Variable, x::Data) = (x, v) |> Multiply()

import Base.(.*)
.*(v1::Variable, v2::Variable) = (v1, v2) |> ElemMultiply()
.*(a::Number, v::Variable) = (Variable(a,nothing), v) |> ElemMultiply()
.*(v::Variable, a::Number) = (v, Variable(a,nothing)) |> ElemMultiply()
.*(x::Data, v::Variable) = (x, v) |> ElemMultiply()
.*(v::Variable, x::Data) = (x, v) |> ElemMultiply()

@compat (f::Multiply)(args) = forward(f, args)
function forward!(f::Multiply, v::Variable)
  v.value = v[1].value * v[2].value
  v.backward! = () -> begin
    T = eltype(v)
    typeof(v[1].value) == Int && println(v[1].value)
    hasgrad(v[1]) && gemm!('N', 'T', T(1), v.grad, v[2].value, T(1), v[1].grad)
    hasgrad(v[2]) && gemm!('T', 'N', T(1), v[1].value, v.grad, T(1), v[2].grad)
  end
end

@compat (f::ElemMultiply)(args) = forward(f, args)
function forward!(f::ElemMultiply, v::Variable)
  v.value = v[1].value .* v[2].value
  v.backward! = () -> begin
    T = eltype(v)
    hasgrad(v[1]) && backward!(f, v[2].value, v[1].grad, v.grad)
    hasgrad(v[2]) && backward!(f, v[1].value, v[2].grad, v.grad)
  end
end

function backward!{T,N}(f::ElemMultiply, x2::Array{T,N}, gx1::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(gx1)
    gx1[i] += gy[i] * x2[i]
  end
end
