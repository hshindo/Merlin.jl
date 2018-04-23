import Base.LinAlg.BLAS: axpy!

#=
for (fname,elty) in (
    (:cublasDcopy,:Float64),
    (:cublasScopy,:Float32))
    @eval begin
        function blascopy!(n::Int, x::CuArray{$elty}, incx::Int,
            y::CuArray{$elty}, incy::Int) where $elty
            $fname(handle(x), n, x, incx, y, incy)
            y
        end
    end
end
=#

for (f,T,Ct) in (
    (:(:cublasDaxpy),:Float64,:Cdouble),
    (:(:cublasSaxpy),:Float32,:Cfloat))
    @eval begin
        function axpy!(n::Int, alpha::$T, x::AbstractCuArray{$T}, incx::Int,
            y::AbstractCuArray{$T}, incy::Int)

            @cublas($f, (
                Ptr{Void},Cint,Ptr{$Ct},Ptr{$Ct},Cint,Ptr{$Ct},Cint),
                gethandle(), n, [alpha], x, incx, y, incy)
            y
        end
    end
end

function axpy!(alpha::T, x::AbstractCuArray{T}, y::AbstractCuArray{T}) where T
    length(x) == length(y) || throw(DimensionMismatch())
    axpy!(length(x), alpha, x, 1, y, 1)
end
function axpy!(n::Int, alpha::T, x::AbstractCuArray{T}, y::AbstractCuArray{T}) where T
    axpy!(n, alpha, x, 1, y, 1)
end
