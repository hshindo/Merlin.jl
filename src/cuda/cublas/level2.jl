import Base.LinAlg.BLAS: gemv, gemv!

for (fname, elty) in ((:cublasDgemv,:Float64), (:cublasSgemv,:Float32))
    @eval begin
        function gemv!(tA::Char, alpha::$elty, A::CuMatrix{$elty}, x::CuVector{$elty},
            beta::$elty, Y::CuVector{$elty})

            @assert device(A) == device(x) == device(Y)
            m, n = size(A)
            length(x) == (tA == 'N' ? n : m) && length(Y) == (tA == 'N' ? m : n) || throw(DimensionMismatch(""))
            $fname(handle(Y), cublasop(tA), m, n,
                $elty[alpha], A, stride(A,2), x, stride(x,1), $elty[beta], Y, stride(Y,1))
            Y
        end
    end
end
function gemv{T}(tA::Char, alpha::T, A::CuMatrix{T}, x::CuVector{T})
    Y = similar(A, size(A, tA=='N' ? 1 : 2))
    gemv!(tA, alpha, A, x, T(0), Y)
end
