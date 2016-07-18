export gemm

function gemm(tA, tB, alpha, A::Var, B::Var)
    (hasdata(A) && hasdata(B)) || return GEMM(nothing, nothing, [A,B], tA, tB, alpha)
    T = eltype(A.data)
    C = BLAS.gemm(tA, tB, T(alpha), A.data, B.data)
    GEMM(C, nothing, [A,B], tA, tB, alpha)
end

type GEMM <: Var
    data
    grad
    tails::Vector
    tA::Char
    tB::Char
    alpha::Float64
end

@compat (v::GEMM)(A::Var, B::Var) = gemm(v.tA, v.tB, v.alpha, A, B)

function backward!(C::GEMM)
    T = eltype(C.data)
    alpha = T(C.alpha)
    A, B = C[1], C[2]
    tA, tB = C.tA, C.tB
    tAt = tA == 'N' ? 'T' : 'N'
    tBt = tB == 'N' ? 'T' : 'N'
    if tA == 'N'
        hasgrad(A) && BLAS.gemm!('N', tBt, alpha, C.grad, B.data, T(1), A.grad)
    else
        hasgrad(A) && BLAS.gemm!(tB, 'T', alpha, B.data, C.grad, T(1), A.grad)
    end
    if tB == 'N'
        hasgrad(B) && BLAS.gemm!(tAt, 'N', alpha, A.data, C.grad, T(1), B.grad)
    else
        hasgrad(B) && BLAS.gemm!('T', tA, alpha, C.grad, A.data, T(1), B.grad)
    end
end
