export gemm, gemm_batch

"""
    gemm(tA::Char, tB::Char, alpha, A::Var, B::Var)
    gemm(A::Var, B::Var; tA='N', tB='N', alpha=1.0)

```math
C = \alpha * \textrm{tA}(A) * \textrm{tB}(B)
```

* tA: transpose A('T') or not('N')
* tB: transpose B('T') or not('N')
"""
function gemm(tA::Char, tB::Char, alpha, A::Var, B::Var)
    (A.data == nothing || B.data == nothing) && return Var(nothing, gemm, (tA,tB,A,B))
    C = BLAS.gemm(tA, tB, eltype(A)(alpha), A.data, B.data)
    function df(gC)
        isconst(A) || ∇gemm_A!(gC, tA, tB, alpha, A.grad, B.data)
        isconst(B) || ∇gemm_B!(gC, tA, tB, alpha, A.data, B.grad)
    end
    Var(C, gemm, (A,B), df)
end
gemm(A::Var, B::Var; tA='N', tB='N', alpha=1.0) = gemm(tA, tB, alpha, A, B)

function ∇gemm_A!(gC, tA::Char, tB::Char, alpha, gA, B)
    T = eltype(gC)
    if tA == 'N'
        BLAS.gemm!('N', tB=='N'?'T':'N', T(alpha), gC, B, T(1), gA)
    else
        BLAS.gemm!(tB, 'T', T(alpha), B, gC, T(1), gA)
    end
end

function ∇gemm_B!(gC, tA::Char, tB::Char, alpha, A, gB)
    T = eltype(gC)
    if tB == 'N'
        BLAS.gemm!(tA=='N'?'T':'N', 'N', T(alpha), A, gC, T(1), gB)
    else
        BLAS.gemm!('T', tA, T(alpha), gC, A, T(1), gB)
    end
end

"""
    gemm_batch(tA::Char, tB::Char, alpha, As::Vector{Var}, B::Vector{Var})
    gemm_batch(As::Vector{Var}, B::Vector{Var}; tA='N', tB='N', alpha=1.0)
"""
function gemm_batch(tA::Char, tB::Char, alpha, As::Vector{Var}, Bs::Vector{Var})
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

function ∇gemm_batch!(tA, tB, alpha, As::Vector, gAs::Vector, Bs::Vector, gBs::Vector, gC::Array)
    @assert length(As) == length(Bs)
    for i = 1:length(As)
        g = view(gC, :, :, i)
        ∇gemm_A!(tB, alpha, gAs[i], Bs[i], g)
        ∇gemm_B!(tA, alpha, As[i], gBs[i], g)
    end
end
∇gemm_batch!(As, gAs, Bs, gBs, gC; tA='N', tB='N', alpha=1.0) = ∇gemm_batch!(tA, tB, alpha, As, gAs, Bs, gBs, gC)
