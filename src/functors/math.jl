"""
*** NOT READY ***
"""
type Add <: Functor
end

import Base.+
+(v1::Variable, v2::Variable) = (v1, v2) |> Add()

function apply(f::Add, x1::Array{T,N}, x2::Array{T,N})
  x = length(x1) > length(x2) ? x1 : x2
  y = similar(x)
  broadcast!(+, y, x1, x2)
  y, gy -> gy, gy -> gy
end

function diff2!{T,N}(f::ElemAdd, x::Array{T,N}, gy::Array{T,N})
  broadcast!(+, output, f.bias, output)
  f.theta.grad += x
  gy
  #sum!(gx1, gy)
  #sum!(gx2, gy)
end

type Multiply <: Functor
end

import Base.*
*(v1::Variable, v2::Variable) = (v1, v2) |> Multiply()

function apply{T}(f::Multiply, x1::Matrix{T}, x2::Matrix{T})
  y = Array(T, size(x1, 1), size(x2, 2))
  gemm!('N', 'N', T(1), x1, x2, T(0), y)
  y, (gy, gx1, gx2) -> diff!(f, x1, x2, gy, gx1, gx2)
end

function diff!{T}(f::Multiply, x1::Matrix{T}, x2::Matrix{T}, gy::Matrix{T}, gx1::Matrix{T}, gx2::Matrix{T})
  gemm!('T', 'N', T(1), x1, gy, T(1), gx2)
  gemm!('N', 'T', T(1), gy, x2, T(1), gx1)
end
