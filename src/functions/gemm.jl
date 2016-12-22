export gemm, gemm_batch

"""
    gemm(tA::Char, tB::Char, alpha, A::Var, B::Var)
    gemm(A::Var, B::Var; tA='N', tB='N', alpha=1.0)

```math
C = \alpha * \textrm{tA}(A) * \textrm{tB}(B)
```

* tA: 'T' (transpose) or 'N' (not transpose)
* tB: same as tA
"""
gemm(tA::Char, tB::Char, alpha, A::Var{Void}, B::Var) = Var(Void(), gemm, (tA,tB,alpha,A,B))
gemm(tA::Char, tB::Char, alpha, A::Var, B::Var{Void}) = Var(Void(), gemm, (tA,tB,alpha,A,B))

function gemm{T}(tA::Char, tB::Char, alpha::T, A::Var, B::Var)
    C = BLAS.gemm(tA, tB, alpha, A.data, B.data)
    df(gC) = ∇gemm!(gC, tA, tB, alpha, A, B)
    Var(C, df, (A,B))
end
gemm(A::Var, B::Var; tA='N', tB='N', alpha=1.0) = gemm(tA, tB, eltype(A.data)(alpha), A, B)

function ∇gemm!{T}(gC, tA::Char, tB::Char, alpha::T, A::Var, B::Var)
    if !isvoid(A.grad)
        if tA == 'N'
            BLAS.gemm!('N', tB=='N'?'T':'N', alpha, gC, B.data, T(1), A.grad)
        else
            BLAS.gemm!(tB, 'T', alpha, B.data, gC, T(1), A.grad)
        end
    end
    if !isvoid(B.grad)
        if tB == 'N'
            BLAS.gemm!(tA=='N'?'T':'N', 'N', alpha, A.data, gC, T(1), B.grad)
        else
            BLAS.gemm!('T', tA, alpha, gC, A.data, T(1), B.grad)
        end
    end
end

"""
    gemm_batch(tA::Char, tB::Char, alpha, As::Vector{Var}, B::Vector{Var})
    gemm_batch(As::Vector{Var}, B::Vector{Var}; tA='N', tB='N', alpha=1.0)

Returns batched gemm.

TODO: add equation here.
"""
gemm_batch(tA::Char, tB::Char, alpha, A::Var{Void}, B::Var) = Var(Void(), gemm_batch, (tA,tB,alpha,A,B))
gemm_batch(tA::Char, tB::Char, alpha, A::Var, B::Var{Void}) = Var(Void(), gemm_batch, (tA,tB,alpha,A,B))

function gemm_batch{T<:Array}(tA::Char, tB::Char, alpha, As::Vector{Var{T}}, Bs::Vector)
    length(As) == length(Bs) || throw(DimensionMismatch("Length of As and Bs must be the same."))

    rowC = tA == 'N' ? size(As[1],1) : size(As[1],2)
    colC = tB == 'N' ? size(Bs[1],2) : size(Bs[1],1)
    C = Array{T}(rowC, colC, length(As))
    for i = 1:length(As)
        BLAS.gemm!(tA, tB, alpha, As[i], Bs[i], T(0), view(C,:,:,i))
    end
    df(gC) = ∇gemm_batch!(gC, tA, tB, alpha, As, Bs)
    Var(C, df, (As,Bs))
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
