module CUBLAS

using ..CUDA
import ..CUDA: devcount, setdevice
import Base.LinAlg.BLAS: axpy!, gemv, gemv!, gemm, gemm!

if is_windows()
    const libcublas = Libdl.find_library(["cublas64_80","cublas64_75"])
else
    const libcublas = Libdl.find_library(["libcublas"])
end
isempty(libcublas) && error("CUBLAS library cannot be found.")

include("lib/$(CUDA.major).$(CUDA.minor)/libcublas.jl")
include("lib/$(CUDA.major).$(CUDA.minor)/libcublas_types.jl")

function check_cublasstatus(status)
    status == CUBLAS_STATUS_SUCCESS && return nothing
    warn("CUBLAS error triggered from:")
    Base.show_backtrace(STDOUT, backtrace())
    throw(errorstring(status))
end

##### Initialization #####
const handles = Ptr{Void}[]
handle(x::CudaArray) = handles[device(x)+1]
atexit(() -> foreach(cublasDestroy,handles))

function init()
    isempty(handles) || throw("Handle is not empty.")
    for dev = 0:devcount()-1
        setdevice(dev)
        p = Ptr{Void}[0]
        cublasCreate(p)
        push!(handles, p[1])
    end
    setdevice(0)
end
init()

function errorstring(status)
    status == CUBLAS_STATUS_SUCCESS && return "SUCCESS"
    status == CUBLAS_STATUS_NOT_INITIALIZED && return "NOT_INITIALIZED"
    status == CUBLAS_STATUS_ALLOC_FAILED && return "ALLOC_FAILED"
    status == CUBLAS_STATUS_INVALID_VALUE && return "INVALID_VALUE"
    status == CUBLAS_STATUS_ARCH_MISMATCH && return "ARCH_MISMATCH"
    status == CUBLAS_STATUS_MAPPING_ERROR && return "MAPPING_ERROR"
    status == CUBLAS_STATUS_EXECUTION_FAILED && return "EXECUTION_FAILED"
    status == CUBLAS_STATUS_INTERNAL_ERROR && return "INTERNAL_ERROR"
    status == CUBLAS_STATUS_NOT_SUPPORTED && return "NOT_SUPPORTED"
    status == CUBLAS_STATUS_LICENSE_ERROR && return "LICENSE_ERROR"
    throw("UNKNOWN ERROR")
end

function cublasop(t::Char)
    t == 'N' && return CUBLAS_OP_N
    t == 'T' && return CUBLAS_OP_T
    t == 'C' && return CUBLAS_OP_C
    throw("Unknown cublas operation: $(t).")
end

##### level1 #####
for (fname,elty) in ((:cublasDcopy,:Float64), (:cublasScopy,:Float32))
    @eval begin
        function blascopy!(n::Int, x::AbstractCudaArray{$elty}, incx::Int,
            y::AbstractCudaArray{$elty}, incy::Int)
            $fname(handle(x), n, x, incx, y, incy)
            y
        end
    end
end

for (fname,elty) in ((:cublasDaxpy,:Float64), (:cublasSaxpy,:Float32))
    @eval begin
        function axpy!(n::Int, alpha::$elty, dx::AbstractCudaArray{$elty}, incx::Int,
            dy::AbstractCudaArray{$elty}, incy::Int)
            $fname(handle(dx), n, [alpha], dx, incx, dy, incy)
            dy
        end
    end
end
function axpy!{T}(alpha::T, x::CudaArray{T}, y::CudaArray{T})
    length(x) == length(y) || throw(DimensionMismatch())
    axpy!(length(x), alpha, x, 1, y, 1)
end
function axpy!{T}(alpha::T, x::CudaArray{T}, rx::Range{Int}, y::CudaArray{T}, ry::Range{Int})
    length(rx) == length(ry) || throw(DimensionMismatch())
    (minimum(rx) < 1 || maximum(rx) > length(x)) && throw(BoundsError())
    (minimum(ry) < 1 || maximum(ry) > length(y)) && throw(BoundsError())
    axpy!(length(rx), alpha, pointer(x,first(rx)-1), step(rx), pointer(y,first(ry)-1), step(ry))
end

##### level2 #####

##### level3 #####
for (fname, elty) in ((:cublasDgemm,:Float64), (:cublasSgemm,:Float32))
    @eval begin
        function gemm!(tA::Char, tB::Char,
            alpha::$elty, A::CudaVecOrMat{$elty}, B::CudaVecOrMat{$elty},
            beta::$elty, C::CudaVecOrMat{$elty})

            @assert device(A) == device(B) == device(C)
            m = size(A, tA == 'N' ? 1 : 2)
            k = size(A, tA == 'N' ? 2 : 1)
            n = size(B, tB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2) || k != size(B, tB == 'N' ? 1 : 2)
                throw(DimensionMismatch())
            end
            $fname(handle(C), cublasop(tA), cublasop(tB), m, n, k,
                $elty[alpha], A, stride(A,2), B, stride(B,2), $elty[beta], C, stride(C,2))
            C
        end
    end
end
function gemm{T}(tA::Char, tB::Char, alpha::T, A::CudaVecOrMat{T}, B::CudaVecOrMat{T})
    C = similar(B, size(A, tA=='N' ? 1 : 2), size(B, tB=='N' ? 2 : 1))
    gemm!(tA, tB, alpha, A, B, T(0), C)
end
function gemm{T}(tA::Char, tB::Char, A::CudaVecOrMat{T}, B::CudaVecOrMat{T})
    gemm(tA, tB, T(1), A, B)
end
gemm{T}(A::CudaVecOrMat{T}, B::CudaVecOrMat{T}; tA='N', tB='N') = gemm(tA, tB, A, B)

for (fname,elty) in [(:cublasDgemmBatched,:Float64), (:cublasSgemmBatched,:Float32)]
    @eval begin
        function gemm_batched!(tA::Char, tB::Char,
            alpha::$elty, As::Vector{CudaMatrix{$elty}}, Bs::Vector{CudaMatrix{$elty}},
            beta::$elty, Cs::Vector{CudaMatrix{$elty}})

            if (length(As) != length(Bs) || length(As) != length(Cs))
                throw(DimensionMismatch(""))
            end
            for i = 1:length(As)
                A, B, C = As[i], Bs[i], Cs[i]
                m = size(A, tA == 'N' ? 1 : 2)
                k = size(A, tA == 'N' ? 2 : 1)
                n = size(B, tB == 'N' ? 2 : 1)
                if m != size(C,1) || n != size(C,2) || k != size(B, tB == 'N' ? 1 : 2)
                    throw(DimensionMismatch(""))
                end
            end
            m = size(As[1], tA == 'N' ? 1 : 2)
            k = size(As[1], tA == 'N' ? 2 : 1)
            n = size(Bs[1], tB == 'N' ? 2 : 1)
            lda = max(1, stride(As[1],2))
            ldb = max(1, stride(Bs[1],2))
            ldc = max(1, stride(Cs[1],2))
            Aptrs = map(a -> Ptr{$elty}(a.ptr), As)
            Bptrs = map(a -> Ptr{$elty}(a.ptr), Bs)
            Cptrs = map(a -> Ptr{$elty}(a.ptr), Cs)
            $fname(handle(C), cublasop(tA), cublasop(tB), m, n, k, [alpha], pointer(Aptrs),
                lda, pointer(Bptrs), ldb, [beta], pointer(Cptrs), ldc, length(As))
            Cs
        end
    end
end
function gemm_batched{T}(tA::Char, tB::Char,
    alpha::T, A::Vector{CudaVecOrMat{T}}, B::Vector{CudaVecOrMat{T}})
    C = CudaMatrix{T}[similar(B[1], (size(A[1], tA=='N' ? 1 : 2), size(B[1], tB=='N' ? 2 : 1))) for i in 1:length(A)]
    gemm_batched!(tA, tB, alpha, A, B, T(0), C)
end
function gemm_batched{T}(tA::Char, tB::Char,
    A::Vector{CudaVecOrMat{T}}, B::Vector{CudaVecOrMat{T}})
    gemm_batched(tA, tB, T(1), A, B)
end

end
