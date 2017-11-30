"""
    gemm(tA::Char, tB::Char, alpha, A::Var, B::Var)
    gemm(A::Var, B::Var, [tA='N'], [tB='N'], [alpha=1])

* tA: 'T' (transpose) or 'N' (not transpose)
* tB: same as tA

```math
C = \alpha \times \textrm{tA}(A) \times \textrm{tB}(B)
```
"""
function BLAS.gemm(tA::Char, tB::Char, alpha::Number, A::Var, B::Var)
    T = eltype(A)
    if nbatchdims(A) == 1
        y = BLAS.gemm(tA, tB, T(alpha), A.data, B.data)
        Var(y, B.batchdims, gemm, (tA,tB,alpha,A,B))
    elseif nbatchdims(B) == 1
        throw("Not implemented yet.")
    else
        throw("batchdims error.")
    end
end

function addgrad!(C::Var, ::typeof(gemm), tA::Char, tB::Char, alpha::Number, A::Var, B::Var)
    T = eltype(C.data)
    if !isvoid(A.grad)
        tA == 'N' ?
        BLAS.gemm!('N', tB=='N'?'T':'N', T(alpha), C.grad, B.data, T(1), A.grad) :
        BLAS.gemm!(tB, 'T', T(alpha), B.data, C.grad, T(1), A.grad)
    end
    if !isvoid(B.grad)
        tB == 'N' ?
        BLAS.gemm!(tA=='N'?'T':'N', 'N', T(alpha), A.data, C.grad, T(1), B.grad) :
        BLAS.gemm!('T', tA, T(alpha), C.grad, A.data, T(1), B.grad)
    end
end
