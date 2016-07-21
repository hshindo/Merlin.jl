export gemm

"""
    gemm(tA, tB, alpha, A, B)

C = alpha * tA(A) * tB(B)

* tA::Char: transpose('T') or not('N'). default: 'N'
* tB::Char: transpose B or not('N'). default: 'N'
* alpha::Float64
"""
function gemm(tA, tB, alpha, A::Var, B::Var)
    hasdata(A,B) || return GEMM(nothing, nothing, [A,B], tA, tB, alpha)
    C = gemm(tA, tB, alpha, A.data, B.data)
    GEMM(C, nothing, [A,B], tA, tB, alpha)
end
gemm(A, B; tA='N', tB='N', alpha=1.0) = gemm(tA, tB, alpha, A, B)

function ∇gemm!(A, gA, B, gB, C, gC; tA='N', tB='N', alpha=1.0)
    ∇gemm!(tA, tB, alpha, A, gA, B, gB, C, gC)
end

type GEMM <: Var
    data
    grad
    tails::Vector
    tA::Char
    tB::Char
    alpha::Float64
end

@compat (C::GEMM)(A::Var, B::Var) = gemm(C.tA, C.tB, C.alpha, A, B)

function gemm{T}(tA, tB, alpha, A::Array{T}, B::Array{T})
    NA, NB = ndims(A), ndims(B)
    @assert 2 <= NA <= 3 && 2 <= NB <= 3
    rowC = tA == 'N' ? size(A,1) : size(A,2)
    colC = tB == 'N' ? size(B,2) : size(B,1)
    nbatches = max(size(A,3),size(B,3))
    C = (NA == 2 && NB == 2) ? Array(T,rowC,colC) : Array(T,rowC,colC,nbatches)
    for i = 1:nbatches
        a = NA == 2 ? A : view(A,:,:,i)
        b = NB == 2 ? B : view(B,:,:,i)
        c = ndims(C) == 2 ? C : view(C,:,:,i)
        BLAS.gemm!(tA, tB, T(alpha), a, b, T(0), c)
    end
    C
end

function backward!(C::GEMM)
    ∇gemm!(C.tA, C.tB, C.alpha, C[1].data, C[1].grad, C[2].data, C[2].grad, C.data, C.grad)
end

function ∇gemm!{T}(tA, tB, alpha, A::Array{T}, gA, B::Array{T}, gB,
    C::Array{T}, gC::Array{T})

    for i = 1:size(C,3)
        a = ndims(A) == 2 ? A : view(A,:,:,i)
        b = ndims(B) == 2 ? B : view(B,:,:,i)
        c = ndims(C) == 2 ? C : view(C,:,:,i)
        gc = ndims(C) == 2 ? gC : view(gC,:,:,i)
        if gA != nothing
            ga = ndims(A) == 2 ? gA : view(gA,:,:,i)
            if tA == 'N'
                t = tB == 'N' ? 'T' : 'N'
                BLAS.gemm!('N', t, T(alpha), gc, b, T(1), ga)
            else
                BLAS.gemm!(tB, 'T', T(alpha), b, gc, T(1), ga)
            end
        end
        if gB != nothing
            gb = ndims(B) == 2 ? gB : view(gB,:,:,i)
            if tB == 'N'
                t = tA == 'N' ? 'T' : 'N'
                BLAS.gemm!(t, 'N', T(alpha), a, gc, T(1), gb)
            else
                BLAS.gemm!('T', tA, T(alpha), gc, a, T(1), gb)
            end
        end
    end
end
