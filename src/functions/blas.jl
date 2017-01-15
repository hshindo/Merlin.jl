export gemv, gemm, gemm_batch

"""
    gemv(tA::Char, alpha, A::Var, x::Var)

* tA: 'T' (transpose) or 'N' (not transpose)

```math
y = \alpha \times \textrm{tA}(A) \times x
```
"""
gemv(tA, alpha, A::Var, x::Var) = forward(gemv, tA, alpha, A, x)
gemv(A::Var, x::Var; tA='N', alpha=1) = gemv(tA, alpha, A, x)

function forward{T}(::typeof(gemv), tA::Char, alpha, A::Matrix{T}, x::Vector{T})
    y = BLAS.gemv(tA, T(alpha), A, x)
    function backward!(gy, tA, alpha, gA, gx)
        isvoid(gA) || BLAS.gemm!('N', 'N', T(alpha), gy, redim(x,2,pad=1), T(1), gA)
        isvoid(gx) || BLAS.gemv!('T', T(alpha), A, gy, T(1), gx)
    end
    y, backward!
end

"""
    gemm(tA::Char, tB::Char, alpha, A::Var, B::Var)
    gemm(A::Var, B::Var, [tA='N'], [tB='N'], [alpha=1])

* tA: 'T' (transpose) or 'N' (not transpose)
* tB: same as tA

```math
C = \alpha \times \textrm{tA}(A) \times \textrm{tB}(B)
```
"""
gemm(tA, tB, alpha, A::Var, B::Var) = forward(gemm, tA, tB, alpha, A, B)
gemm(A::Var, B::Var; tA='N', tB='N', alpha=1) = gemm(tA, tB, alpha, A, B)

function forward{T}(::typeof(gemm), tA::Char, tB::Char, alpha, A::Matrix{T}, B::Matrix{T})
    C = BLAS.gemm(tA, tB, T(alpha), A, B)
    function backward!(gC, tA, tB, alpha, gA, gB)
        if !isvoid(gA)
            tA == 'N' ?
            BLAS.gemm!('N', tB=='N'?'T':'N', T(alpha), gC, B, T(1), gA) :
            BLAS.gemm!(tB, 'T', T(alpha), B, gC, T(1), gA)
        end
        if !isvoid(gB)
            tB == 'N' ?
            BLAS.gemm!(tA=='N'?'T':'N', 'N', T(alpha), A, gC, T(1), gB) :
            BLAS.gemm!('T', tA, T(alpha), gC, A, T(1), gB)
        end
    end
    C, backward!
end

"""
    gemm_batch(tA::Char, tB::Char, alpha, As::Vector{Var}, B::Vector{Var})
    gemm_batch(As::Vector{Var}, B::Vector{Var}, [tA='N'], [tB='N'], [alpha=1])
"""
gemm_batch(tA, tB, alpha, As::Vector{Var}, Bs::Vector{Var}) = forward(tA, tB, alpha, As, Bs)
gemm_batch(As, Bs; tA='N', tB='N', alpha=1.0) = gemm_batch(tA, tB, alpha, As, Bs)

function forward{T}(::typeof(gemm_batch), tA::Char, tB::Char, alpha, As::Vector{Matrix{T}}, Bs::Vector{Matrix{T}})
    length(As) == length(Bs) || throw(DimensionMismatch("Length of As and Bs must be the same."))

    rowC = tA == 'N' ? size(As[1],1) : size(As[1],2)
    colC = tB == 'N' ? size(Bs[1],2) : size(Bs[1],1)
    T = eltype(As[1])
    C = Array{T}(rowC, colC, length(As))
    for i = 1:length(As)
        BLAS.gemm!(tA, tB, alpha, As[i], Bs[i], T(0), view(C,:,:,i))
    end
    df(gC) = ∇gemm_batch!(gC, tA, tB, alpha, As, Bs)
    Var(C, df, (As,Bs))
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
