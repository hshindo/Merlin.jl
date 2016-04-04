type Multiply <: Functor
end

function call(f::Multiply, arg1::Variable, arg2::Variable)
  v.value = arg1.value * arg2.value
  v.backward! = () -> begin
    T = eltype(v.value)
    v[1].grad == nothing && (v[1].grad = zeros(v[1].value))
    v[2].grad == nothing && (v[2].grad = zeros(v[2].value))
    gemm!('N', 'T', T(1), v.grad, v[2].value, T(1), v[1].grad)
    gemm!('T', 'N', T(1), v[1].value, v.grad, T(1), v[2].grad)
  end
end

function multiply!(f::Multiply, x1::Array, x2::Array, y::Array)
  gemm!('N', 'N', T(1), x1, x2, T(-1), y)
end

import Base.*
*(v1::Variable, v2::Variable) = Multiply()(v1, v2)

type Add <: Functor
  inplace::Bool
end

Add() = Add(false)

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

function call(f::Add, arg1::Variable, arg2::Variable)
  x1, x2 = arg1.value, arg2.value
  y = length(x1) >= length(x2) ? x1 : x2
  broadcast!(+, y, x1, x2)
  Variable()
end

function forward!(f::Add, v::Variable)
  v.value = broadcast(+, v[1].value, v[2].value)
  v.backward! = () -> begin
    if v[1].grad == nothing
      if length(v[1].value) == length(v.value)
        v[1].grad = v.grad
      else
        error("unexpected1")
      end
    else
      ∇add!(v[1].grad, v.grad)
    end
    if v[2].grad == nothing
      if length(v[2].value) == length(v.value)
        v[2].grad = v.grad
      else
        error("unexpected1")
      end
    else
      ∇add!(v[2].grad, v.grad)
    end
  end
end

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

"""
## GEMM
BLAS gemm function.
This is in-place operation.
"""
type GEMM <: Functor
end

function call(f::GEMM, y::Variable, x1::Variable, x2::Variable)
  T = eltype(y)
  gemm!('N', 'N', T(1), x1.value, x2.value, T(1), y.value)
  backward! = gy -> begin
    ()
  end
  Variable(f, [y,x1,x2], y.value, backward!)
end
