import LinearAlgebra.BLAS: scal!, axpy!

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
        function axpy!(n::Int, alpha::$T, x::Union{CuArray{$T},CuPtr{$T}}, incx::Int,
            y::Union{CuArray{$T},CuPtr{$T}}, incy::Int)

            @cublas($f, (
                Ptr{Cvoid},Cint,Ptr{$Ct},Ptr{$Ct},Cint,Ptr{$Ct},Cint),
                gethandle(), n, [alpha], x, incx, y, incy)
            y
        end
    end
end

function axpy!(n::Int, alpha::T, x::Union{CuArray{T},CuPtr{T}}, y::Union{CuArray{T},CuPtr{T}}) where T
    axpy!(n, alpha, x, 1, y, 1)
end

function axpy!(alpha::T, x::CuArray{T}, y::CuArray{T}) where T
    length(x) == length(y) || throw(DimensionMismatch())
    axpy!(length(x), alpha, x, y)
end
