type Add <: Functor
end

import Base.+
+(v1::Var, v2::Var) = (v1, v2) |> Add()

function forward{T,N}(f::Add, x1::Array{T,N}, x2::Array{T,N})
  x = length(x1) > length(x2) ? x1 : x2
  y = similar(x)
  broadcast!(+, y, x1, x2)
  y, (gy, gx1, gx2) -> begin
    gx1 == nothing || broadcast!(+, gx1, gx1, gy)
    gx2 == nothing || broadcast!(+, gx2, gx2, gy)
  end
end

type Multiply <: Functor
end

import Base.*
*(v1::Var, v2::Var) = (v1, v2) |> Multiply()

function forward{T}(f::Multiply, x1::Matrix{T}, x2::Matrix{T})
  y = Array(T, size(x1, 1), size(x2, 2))
  gemm!('N', 'N', T(1), x1, x2, T(0), y)
  function backward!(gy, gx1, gx2)
    gx1 == nothing || gemm!('T', 'N', T(1), x1, gy, T(1), gx2)
    gx2 == nothing || gemm!('N', 'T', T(1), gy, x2, T(1), gx1)
  end
  y, backward!
end
