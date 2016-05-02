export Add, ElemAdd
export Subtract, ElemSubtract
export Multiply, ElemMultiply

type Add <: Functor
end
type ElemAdd <: Functor
end
type Subtract <: Functor
end
type ElemSubtract <: Functor
end
type Multiply <: Functor
end
type ElemMultiply <: Functor
end

import Base.+
+(v1::Variable, v2::Variable) = (v1, v2) |> Add()
+(a::Number, v::Variable) = (Variable(a,nothing), v) |> Add()
+(v::Variable, a::Number) = a + v
+(x::Data, v::Variable) = (x, v) |> Add()
+(v::Variable, x::Data) = (x, v) |> Add()

import Base.(.+)
.+(v1::Variable, v2::Variable) = (v1, v2) |> ElemAdd()
.+(x::Data, v::Variable) = (x, v) |> ElemAdd()
.+(v::Variable, x::Data) = (x, v) |> ElemAdd()

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

@compat (f::Add)(args) = forward(f, args)
function forward!(f::Add, v::Variable)
  v.value = v[1].value + v[2].value
  v.backward! = () -> begin
    T = eltype(v)
    hasgrad(v[1]) && BLAS.axpy!(T(1), v.grad, v[1].grad)
    hasgrad(v[2]) && BLAS.axpy!(T(1), v.grad, v[2].grad)
  end
end

@compat (f::ElemAdd)(args) = forward(f, args)
function forward!(f::ElemAdd, v::Variable)
  v.value = v[1].value .+ v[2].value
  v.backward! = () -> begin
    hasgrad(v[1]) && backward!(f, v[1].grad, v.grad)
    hasgrad(v[2]) && backward!(f, v[2].grad, v.grad)
  end
end

function backward!{T,N}(f::ElemAdd, gx::Array{T,N}, gy::Array{T,N})
  for offset = 1:length(gx):length(gy)
    BLAS.axpy!(length(gx), T(1), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
  end
end

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

@compat (f::Multiply)(args) = forward(f, args)
function forward!(f::Multiply, v::Variable)
  v.value = v[1].value * v[2].value
  v.backward! = () -> begin
    T = eltype(v)
    typeof(v[1].value) == Int && println(v[1].value)
    hasgrad(v[1]) && BLAS.gemm!('N', 'T', T(1), v.grad, v[2].value, T(1), v[1].grad)
    hasgrad(v[2]) && BLAS.gemm!('T', 'N', T(1), v[1].value, v.grad, T(1), v[2].grad)
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
