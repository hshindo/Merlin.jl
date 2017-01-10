export gemv, gemm, gemm_batch

"""
    gemv(tA::Char, alpha, A::Var, x::Var)

* tA: 'T' (transpose) or 'N' (not transpose)

```math
y = \alpha \times \textrm{tA}(A) \times x
```
"""
function gemv{T}(tA::Char, alpha::T, A::Var, x::Var)
    (isa(A.data,Void) || isa(x.data,Void)) && return Var(nothing, gemv, (tA,A,x))

    y = BLAS.gemv(tA, alpha, A.data, x.data)
    function df(gy)
        isa(A.grad, Void) || BLAS.gemm!('N', 'N', alpha, gy, redim(x.data,2,pad=1), T(1), A.grad)
        isa(x.grad, Void) || BLAS.gemv!('T', alpha, A.data, gy, T(1), x.grad)
    end
    Var(y, df, (A,x))
end
gemv(A::Var, x::Var; tA='N', alpha=1.0) = gemv(tA, eltype(x.data)(alpha), A, x)

"""
    gemm(tA::Char, tB::Char, alpha, A::Var, B::Var)
    gemm(A::Var, B::Var, [tA='N'], [tB='N'], [alpha=1])

* tA: 'T' (transpose) or 'N' (not transpose)
* tB: same as tA

```math
C = \alpha \times \textrm{tA}(A) \times \textrm{tB}(B)
```
"""
function gemm(tA::Char, tB::Char, alpha, A::Var, B::Var)
    (isa(A.data,Void) || isa(B.data,Void)) && return Var(nothing, gemm, (tA,tB,alpha,A,B))
    T = eltype(A.data)
    C = BLAS.gemm(tA, tB, T(alpha), A.data, B.data)
    function df(gC)
        if !isa(A.grad, Void)
            tA == 'N' ?
            BLAS.gemm!('N', tB=='N'?'T':'N', T(alpha), gC, B.data, T(1), A.grad) :
            BLAS.gemm!(tB, 'T', T(alpha), B.data, gC, T(1), A.grad)
        end
        if !isa(B.grad, Void)
            tB == 'N' ?
            BLAS.gemm!(tA=='N'?'T':'N', 'N', T(alpha), A.data, gC, T(1), B.grad) :
            BLAS.gemm!('T', tA, T(alpha), gC, A.data, T(1), B.grad)
        end
    end
    Var(C, df, (A,B))
end
gemm(A::Var, B::Var; tA='N', tB='N', alpha=1) = gemm(tA, tB, eltype(A.data)(alpha), A, B)

"""
    gemm_batch(tA::Char, tB::Char, alpha, As::Vector{Var}, B::Vector{Var})
    gemm_batch(As::Vector{Var}, B::Vector{Var}, [tA='N'], [tB='N'], [alpha=1])
"""
function gemm_batch{A<:Array,B<:Array}(tA::Char, tB::Char, alpha, As::Vector{Var{A}}, Bs::Vector{Var{B}})
    length(As) == length(Bs) || throw(DimensionMismatch("Length of As and Bs must be the same."))

    rowC = tA == 'N' ? size(As[1].data,1) : size(As[1].data,2)
    colC = tB == 'N' ? size(Bs[1].data,2) : size(Bs[1].data,1)
    T = eltype(As[1].data)
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
