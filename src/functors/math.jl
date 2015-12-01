using Base.LinAlg.BLAS

type Add <: Functor
end

import Base.+
+(v1::Variable, v2::Variable) = (v1, v2) |> Add()

apply{T,N}(fun::Add, x1::Array{T,N}, x2::Array{T,N}) = x1 + x2

function diff(fun::Add, input::Tuple{Array,Array}, gradout::Array)
  gradout, gradout
end

type Mult <: Functor
end

import Base.*
*(v1::Variable, v2::Variable) = (v1, v2) |> Mult()

function apply{T}(fun::Mult, x1::Matrix{T}, x2::Matrix{T})
  y = Array(T, size(x1, 1), size(x2, 2))
  gemm!('N', 'N', 1.0, x1, x2, 0.0, y)
  y
end

function diff{T}(fun::Mult, inputs::Tuple{Matrix{T},Matrix{T}}, gradout::Matrix{T})
  grad1 = similar(x1)
  grad2 = similar(x2)
  gemm!('T', 'N', 1.0, x1, gradout, 0.0, x2)
  gemm!('N', 'T', 1.0, gradout, x2, 1.0, x1)
  grad1, grad2
end
