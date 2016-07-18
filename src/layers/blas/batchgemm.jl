export batch_gemm

type BatchGEMM <: Var
    data
    grad
    tails::Vector
    tA::Char
    tB::Char
    alpha::Float64
end

function batch_gemm{T}(tA, tB, alpha, A::Array{T,3}, B::Array{T,3})
    C = Array(T, )
    C = BLAS.gemm(tA, tB, T(alpha), A.data, B.data)
    GEMM(C, nothing, [A,B], tA, tB, alpha)
end

function gemm(tA, tB, alpha, A::Var, B::Var)
    (hasdata(A) && hasdata(B)) || return GEMM(nothing, nothing, [A,B], tA, tB, alpha)
    T = eltype(A.data)
    C = BLAS.gemm(tA, tB, T(alpha), A.data, B.data)
    GEMM(C, nothing, [A,B], tA, tB, alpha)
end
