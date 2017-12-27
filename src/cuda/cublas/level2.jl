import Base.LinAlg.BLAS: gemv, gemv!

for (f,T,Ct) in (
    (:(:cublasDgemv),:Float64,:Cdouble),
    (:(:cublasSgemv),:Float32,:Cfloat))
    @eval begin
        function gemv!(tA::Char, alpha::$T, A::CuMatrix{$T}, x::CuVector{$T},
            beta::$T, y::CuVector{$T})

            @assert getdevice(A) == getdevice(x) == getdevice(y)
            m, n = size(A)
            length(x) == (tA == 'N' ? n : m) || throw(DimensionMismatch(""))
            length(y) == (tA == 'N' ? m : n) || throw(DimensionMismatch(""))
            h = handle(x)
            lda = max(1, stride(A,2))
            incx = stride(x, 1)
            incy = stride(y, 1)
            @apicall($f, (
                cublasHandle_t,Cint,Cint,Cint,
                Ptr{$Ct},Ptr{$Ct},Cint,
                Ptr{$Ct},Cint,
                Ptr{$Ct},Ptr{$Ct},Cint),
                h, cublasop(tA), m, n, [alpha], A, lda, x, incx, [beta], y, incy)
            y
        end
    end
end

function gemv(tA::Char, alpha::T, A::CuMatrix{T}, x::CuVector{T}) where T
    y = similar(A, size(A, tA=='N' ? 1 : 2))
    gemv!(tA, alpha, A, x, T(0), y)
end
