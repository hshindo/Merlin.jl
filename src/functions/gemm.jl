export gemm, gemm_batch

"""
    gemm(tA::Char, tB::Char, alpha::Float64, A::Var, B::Var)
    gemm(A::Var, B::Var; tA='N', tB='N', alpha=1.0)

```math
C=\alpha*\textrm{tA}(A)*\textrm{tB}(B)
```

* tA: transpose('T') or not('N')
* tB: transpose('T') or not('N')
"""
@graph function gemm(tA, tB, alpha, A::Var, B::Var)
    C = BLAS.gemm(tA, tB, alpha, A.data, B.data)
    function df(gC)
        ∇gemm_A!(tB, alpha, A.grad, B, gC)
        ∇gemm_B!(tA, alpha, A, B.grad, gC)
    end
    Var(C, [A,B], gemm, df)
end
gemm(A, B; tA='N', tB='N', alpha=1.0) = gemm(tA, tB, alpha, A, B)

function gemm_batch(tA, tB, alpha, As::Vector, Bs::Vector)
    @assert length(As) == length(Bs)
    T = eltype(As[1])
    rowC = tA == 'N' ? size(As[1],1) : size(As[1],2)
    colC = tB == 'N' ? size(Bs[1],2) : size(Bs[1],1)
    C = similar(As[1], rowC, colC, length(As))
    for i = 1:length(As)
        BLAS.gemm!(tA, tB, T(alpha), As[i], Bs[i], T(0), view(C,:,:,i))
    end
    C
end
gemm_batch(As, Bs; tA='N', tB='N', alpha=1.0) = gemm_batch(tA, tB, alpha, As, Bs)

function ∇gemm_A!(tA::Char, tB::Char, alpha, gA, B, gC)
    T = eltype(gC)
    if tA == 'N'
        BLAS.gemm!('N', tB=='N'?'T':'N', T(alpha), gC, B, T(1), gA)
    else
        BLAS.gemm!(tB, 'T', T(alpha), B, gC, T(1), gA)
    end
end

function ∇gemm_B!(tA::Char, tB::Char, alpha, A, gB, gC)
    T = eltype(gC)
    if tB == 'N'
        BLAS.gemm!(tA=='N'?'T':'N', 'N', T(alpha), A, gC, T(1), gB)
    else
        BLAS.gemm!('T', tA, T(alpha), gC, A, T(1), gB)
    end
end

function ∇gemm_batch!(tA, tB, alpha, As::Vector, gAs::Vector, Bs::Vector, gBs::Vector, gC::Array)
    @assert length(As) == length(Bs)
    for i = 1:length(As)
        g = view(gC, :, :, i)
        ∇gemm_A!(tB, alpha, gAs[i], Bs[i], g)
        ∇gemm_B!(tA, alpha, As[i], gBs[i], g)
    end
end
∇gemm_batch!(As, gAs, Bs, gBs, gC; tA='N', tB='N', alpha=1.0) = ∇gemm_batch!(tA, tB, alpha, As, gAs, Bs, gBs, gC)
