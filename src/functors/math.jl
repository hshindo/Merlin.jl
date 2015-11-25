using Base.LinAlg.BLAS

type Add <: Functor
end

import Base.+
+(v1::Variable, v2::Variable) = (v1, v2) |> Add()

function apply(fun::Add, input::Tuple{Array,Array})
  input[1] + input[2]
end

function diff(fun::Add, input::Tuple{Array,Array}, gradout::Array)
  gradout, gradout
end

type Mult <: Functor
end

import Base.*
*(v1::Variable, v2::Variable) = (v1, v2) |> Mult()

function apply(fun::Mult, input::Tuple{Matrix,Matrix})
  x1, x2 = input
  y = Array(eltype(x1), size(x1, 1), size(x2, 2))
  gemm!('N', 'N', 1.0, x1, x2, 0.0, y)
  y
end

function diff(fun::Mult, inputs::Tuple{Matrix,Matrix}, gradout::Matrix)
  T = eltype(x1)
  grad1 = similar(x1)
  grad2 = similar(x2)
  gemm!('T', 'N', 1.0, x1, gradout, 0.0, x2)
  gemm!('N', 'T', 1.0, gradout, x2, 1.0, x1)
  grad1, grad2
end
