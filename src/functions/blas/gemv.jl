"""
    BLAS.gemv(tA::Char, alpha, A::Var, x::Var)

* tA: 'T' (transpose) or 'N' (not transpose)

```math
y = \alpha \times \textrm{tA}(A) \times x
```
"""
function BLAS.gemv(tA::Char, alpha::Number, A::Var, x::Var)
    T = eltype(A)
    if nbatchdims(A) == 1
        ndims(A) == 2 || throw("ndims(A) must be 2.")
        if nbatchdims(x) == 1
            y = BLAS.gemv(tA, T(alpha), A.data, x.data)
            Var(y, [length(y)], gemv, (tA,alpha,A,x))
        else
            data = reshape(x.data, x.batchdims[1], nbatchdims(x))
            y = BLAS.gemm(tA, 'N', T(alpha), A.data, data)
            batchdims = fill(size(y,1), size(y,2))
            y = reshape(y, length(y))
            Var(y, batchdims, gemv, (tA,alpha,A,x))
        end
    elseif nbatchdims(x) == 1
        ys = map(unsafe_split(A.data,A.batchdims)) do AA
            BLAS.gemv(tA, T(alpha), AA, x.data)
        end
        y = cat(1, ys...)
        batchdims = fill(length(ys[1]), length(ys))
        Var(y, batchdims, gemv, (tA,alpha,A,x))
    else
        throw("Error")
    end
end

function addgrad!(y::Var, ::typeof(gemv), tA::Char, alpha::Number, A::Var, x::Var)
    T = eltype(A.data)
    if !isvoid(A.grad)
        mat(v) = reshape(v, length(v), 1)
        tA == 'N' ?
        BLAS.gemm!('N', 'T', T(alpha), mat(y.grad), mat(x.data), T(1), A.grad) :
        BLAS.gemm!('N', 'T', T(alpha), mat(x.data), mat(y.grad), T(1), A.grad)
    end
    isvoid(x.grad) || BLAS.gemv!(tA=='N'?'T':'N', T(alpha), A.data, y.grad, T(1), x.grad)
end
