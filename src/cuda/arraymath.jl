import Base: *

*(A::CuMatrix{T}, x::CuVector{T}) where T = BLAS.gemv('N', T(1), A, x)
*(A::CuMatrix{T}, B::CuMatrix{T}) where T = BLAS.gemm('N', 'N', T(1), A, B)
