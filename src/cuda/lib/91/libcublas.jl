# Julia wrapper for header: /usr/local/cuda/include/cublas_v2.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function cublasCreate_v2(handle)
    ccall((:cublasCreate_v2, libcublas), cublasStatus_t, (Ptr{cublasHandle_t},), handle)
end

function cublasDestroy_v2(handle)
    ccall((:cublasDestroy_v2, libcublas), cublasStatus_t, (cublasHandle_t,), handle)
end

function cublasGetVersion_v2(handle, version)
    ccall((:cublasGetVersion_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cint}), handle, version)
end

function cublasGetProperty(_type, value)
    ccall((:cublasGetProperty, libcublas), cublasStatus_t, (libraryPropertyType, Ptr{Cint}), _type, value)
end

function cublasSetStream_v2(handle, streamId)
    ccall((:cublasSetStream_v2, libcublas), cublasStatus_t, (cublasHandle_t, cudaStream_t), handle, streamId)
end

function cublasGetStream_v2(handle, streamId)
    ccall((:cublasGetStream_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cudaStream_t}), handle, streamId)
end

function cublasGetPointerMode_v2(handle, mode)
    ccall((:cublasGetPointerMode_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cublasPointerMode_t}), handle, mode)
end

function cublasSetPointerMode_v2(handle, mode)
    ccall((:cublasSetPointerMode_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasPointerMode_t), handle, mode)
end

function cublasGetAtomicsMode(handle, mode)
    ccall((:cublasGetAtomicsMode, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cublasAtomicsMode_t}), handle, mode)
end

function cublasSetAtomicsMode(handle, mode)
    ccall((:cublasSetAtomicsMode, libcublas), cublasStatus_t, (cublasHandle_t, cublasAtomicsMode_t), handle, mode)
end

function cublasGetMathMode(handle, mode)
    ccall((:cublasGetMathMode, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cublasMath_t}), handle, mode)
end

function cublasSetMathMode(handle, mode)
    ccall((:cublasSetMathMode, libcublas), cublasStatus_t, (cublasHandle_t, cublasMath_t), handle, mode)
end

function cublasSetVector(n, elemSize, x, incx, devicePtr, incy)
    ccall((:cublasSetVector, libcublas), cublasStatus_t, (Cint, Cint, Ptr{Void}, Cint, Ptr{Void}, Cint), n, elemSize, x, incx, devicePtr, incy)
end

function cublasGetVector(n, elemSize, x, incx, y, incy)
    ccall((:cublasGetVector, libcublas), cublasStatus_t, (Cint, Cint, Ptr{Void}, Cint, Ptr{Void}, Cint), n, elemSize, x, incx, y, incy)
end

function cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb)
    ccall((:cublasSetMatrix, libcublas), cublasStatus_t, (Cint, Cint, Cint, Ptr{Void}, Cint, Ptr{Void}, Cint), rows, cols, elemSize, A, lda, B, ldb)
end

function cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb)
    ccall((:cublasGetMatrix, libcublas), cublasStatus_t, (Cint, Cint, Cint, Ptr{Void}, Cint, Ptr{Void}, Cint), rows, cols, elemSize, A, lda, B, ldb)
end

function cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream)
    ccall((:cublasSetVectorAsync, libcublas), cublasStatus_t, (Cint, Cint, Ptr{Void}, Cint, Ptr{Void}, Cint, cudaStream_t), n, elemSize, hostPtr, incx, devicePtr, incy, stream)
end

function cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream)
    ccall((:cublasGetVectorAsync, libcublas), cublasStatus_t, (Cint, Cint, Ptr{Void}, Cint, Ptr{Void}, Cint, cudaStream_t), n, elemSize, devicePtr, incx, hostPtr, incy, stream)
end

function cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream)
    ccall((:cublasSetMatrixAsync, libcublas), cublasStatus_t, (Cint, Cint, Cint, Ptr{Void}, Cint, Ptr{Void}, Cint, cudaStream_t), rows, cols, elemSize, A, lda, B, ldb, stream)
end

function cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream)
    ccall((:cublasGetMatrixAsync, libcublas), cublasStatus_t, (Cint, Cint, Cint, Ptr{Void}, Cint, Ptr{Void}, Cint, cudaStream_t), rows, cols, elemSize, A, lda, B, ldb, stream)
end

function cublasXerbla(srName, info)
    ccall((:cublasXerbla, libcublas), Void, (Cstring, Cint), srName, info)
end

function cublasNrm2Ex(handle, n, x, xType, incx, result, resultType, executionType)
    ccall((:cublasNrm2Ex, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Void}, cudaDataType, Cint, Ptr{Void}, cudaDataType, cudaDataType), handle, n, x, xType, incx, result, resultType, executionType)
end

function cublasSnrm2_v2(handle, n, x, incx, result)
    ccall((:cublasSnrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, n, x, incx, result)
end

function cublasDnrm2_v2(handle, n, x, incx, result)
    ccall((:cublasDnrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, n, x, incx, result)
end

function cublasScnrm2_v2(handle, n, x, incx, result)
    ccall((:cublasScnrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}), handle, n, x, incx, result)
end

function cublasDznrm2_v2(handle, n, x, incx, result)
    ccall((:cublasDznrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}), handle, n, x, incx, result)
end

function cublasDotEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
    ccall((:cublasDotEx, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Void}, cudaDataType, Cint, Ptr{Void}, cudaDataType, Cint, Ptr{Void}, cudaDataType, cudaDataType), handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
end

function cublasDotcEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
    ccall((:cublasDotcEx, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Void}, cudaDataType, Cint, Ptr{Void}, cudaDataType, Cint, Ptr{Void}, cudaDataType, cudaDataType), handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
end

function cublasSdot_v2(handle, n, x, incx, y, incy, result)
    ccall((:cublasSdot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, n, x, incx, y, incy, result)
end

function cublasDdot_v2(handle, n, x, incx, y, incy, result)
    ccall((:cublasDdot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, n, x, incx, y, incy, result)
end

function cublasCdotu_v2(handle, n, x, incx, y, incy, result)
    ccall((:cublasCdotu_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}), handle, n, x, incx, y, incy, result)
end

function cublasCdotc_v2(handle, n, x, incx, y, incy, result)
    ccall((:cublasCdotc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}), handle, n, x, incx, y, incy, result)
end

function cublasZdotu_v2(handle, n, x, incx, y, incy, result)
    ccall((:cublasZdotu_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}), handle, n, x, incx, y, incy, result)
end

function cublasZdotc_v2(handle, n, x, incx, y, incy, result)
    ccall((:cublasZdotc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}), handle, n, x, incx, y, incy, result)
end

function cublasScalEx(handle, n, alpha, alphaType, x, xType, incx, executionType)
    ccall((:cublasScalEx, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Void}, cudaDataType, Ptr{Void}, cudaDataType, Cint, cudaDataType), handle, n, alpha, alphaType, x, xType, incx, executionType)
end

function cublasSscal_v2(handle, n, alpha, x, incx)
    ccall((:cublasSscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, n, alpha, x, incx)
end

function cublasDscal_v2(handle, n, alpha, x, incx)
    ccall((:cublasDscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, n, alpha, x, incx)
end

function cublasCscal_v2(handle, n, alpha, x, incx)
    ccall((:cublasCscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, n, alpha, x, incx)
end

function cublasCsscal_v2(handle, n, alpha, x, incx)
    ccall((:cublasCsscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint), handle, n, alpha, x, incx)
end

function cublasZscal_v2(handle, n, alpha, x, incx)
    ccall((:cublasZscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, n, alpha, x, incx)
end

function cublasZdscal_v2(handle, n, alpha, x, incx)
    ccall((:cublasZdscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint), handle, n, alpha, x, incx)
end

function cublasAxpyEx(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype)
    ccall((:cublasAxpyEx, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Void}, cudaDataType, Ptr{Void}, cudaDataType, Cint, Ptr{Void}, cudaDataType, Cint, cudaDataType), handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype)
end

function cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy)
    ccall((:cublasSaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, n, alpha, x, incx, y, incy)
end

function cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy)
    ccall((:cublasDaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, n, alpha, x, incx, y, incy)
end

function cublasCaxpy_v2(handle, n, alpha, x, incx, y, incy)
    ccall((:cublasCaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, n, alpha, x, incx, y, incy)
end

function cublasZaxpy_v2(handle, n, alpha, x, incx, y, incy)
    ccall((:cublasZaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, n, alpha, x, incx, y, incy)
end

function cublasScopy_v2(handle, n, x, incx, y, incy)
    ccall((:cublasScopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, n, x, incx, y, incy)
end

function cublasDcopy_v2(handle, n, x, incx, y, incy)
    ccall((:cublasDcopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, n, x, incx, y, incy)
end

function cublasCcopy_v2(handle, n, x, incx, y, incy)
    ccall((:cublasCcopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, n, x, incx, y, incy)
end

function cublasZcopy_v2(handle, n, x, incx, y, incy)
    ccall((:cublasZcopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, n, x, incx, y, incy)
end

function cublasSswap_v2(handle, n, x, incx, y, incy)
    ccall((:cublasSswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, n, x, incx, y, incy)
end

function cublasDswap_v2(handle, n, x, incx, y, incy)
    ccall((:cublasDswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, n, x, incx, y, incy)
end

function cublasCswap_v2(handle, n, x, incx, y, incy)
    ccall((:cublasCswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, n, x, incx, y, incy)
end

function cublasZswap_v2(handle, n, x, incx, y, incy)
    ccall((:cublasZswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, n, x, incx, y, incy)
end

function cublasIsamax_v2(handle, n, x, incx, result)
    ccall((:cublasIsamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end

function cublasIdamax_v2(handle, n, x, incx, result)
    ccall((:cublasIdamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end

function cublasIcamax_v2(handle, n, x, incx, result)
    ccall((:cublasIcamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end

function cublasIzamax_v2(handle, n, x, incx, result)
    ccall((:cublasIzamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end

function cublasIsamin_v2(handle, n, x, incx, result)
    ccall((:cublasIsamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end

function cublasIdamin_v2(handle, n, x, incx, result)
    ccall((:cublasIdamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end

function cublasIcamin_v2(handle, n, x, incx, result)
    ccall((:cublasIcamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end

function cublasIzamin_v2(handle, n, x, incx, result)
    ccall((:cublasIzamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end

function cublasSasum_v2(handle, n, x, incx, result)
    ccall((:cublasSasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, n, x, incx, result)
end

function cublasDasum_v2(handle, n, x, incx, result)
    ccall((:cublasDasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, n, x, incx, result)
end

function cublasScasum_v2(handle, n, x, incx, result)
    ccall((:cublasScasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}), handle, n, x, incx, result)
end

function cublasDzasum_v2(handle, n, x, incx, result)
    ccall((:cublasDzasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}), handle, n, x, incx, result)
end

function cublasSrot_v2(handle, n, x, incx, y, incy, c, s)
    ccall((:cublasSrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}), handle, n, x, incx, y, incy, c, s)
end

function cublasDrot_v2(handle, n, x, incx, y, incy, c, s)
    ccall((:cublasDrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), handle, n, x, incx, y, incy, c, s)
end

function cublasCrot_v2(handle, n, x, incx, y, incy, c, s)
    ccall((:cublasCrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{cuComplex}), handle, n, x, incx, y, incy, c, s)
end

function cublasCsrot_v2(handle, n, x, incx, y, incy, c, s)
    ccall((:cublasCsrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{Cfloat}), handle, n, x, incx, y, incy, c, s)
end

function cublasZrot_v2(handle, n, x, incx, y, incy, c, s)
    ccall((:cublasZrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}), handle, n, x, incx, y, incy, c, s)
end

function cublasZdrot_v2(handle, n, x, incx, y, incy, c, s)
    ccall((:cublasZdrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), handle, n, x, incx, y, incy, c, s)
end

function cublasSrotg_v2(handle, a, b, c, s)
    ccall((:cublasSrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}), handle, a, b, c, s)
end

function cublasDrotg_v2(handle, a, b, c, s)
    ccall((:cublasDrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), handle, a, b, c, s)
end

function cublasCrotg_v2(handle, a, b, c, s)
    ccall((:cublasCrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cuComplex}, Ptr{cuComplex}, Ptr{Cfloat}, Ptr{cuComplex}), handle, a, b, c, s)
end

function cublasZrotg_v2(handle, a, b, c, s)
    ccall((:cublasZrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Ptr{Cdouble}, Ptr{cuDoubleComplex}), handle, a, b, c, s)
end

function cublasSrotm_v2(handle, n, x, incx, y, incy, param)
    ccall((:cublasSrotm_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, n, x, incx, y, incy, param)
end

function cublasDrotm_v2(handle, n, x, incx, y, incy, param)
    ccall((:cublasDrotm_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, n, x, incx, y, incy, param)
end

function cublasSrotmg_v2(handle, d1, d2, x1, y1, param)
    ccall((:cublasSrotmg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}), handle, d1, d2, x1, y1, param)
end

function cublasDrotmg_v2(handle, d1, d2, x1, y1, param)
    ccall((:cublasDrotmg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), handle, d1, d2, x1, y1, param)
end

function cublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasSgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasDgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasCgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasCgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasZgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasSgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasSgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasDgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasCgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasCgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasZgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasStrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    ccall((:cublasStrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasDtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    ccall((:cublasDtrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasCtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    ccall((:cublasCtrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasZtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    ccall((:cublasZtrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasStbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    ccall((:cublasStbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasDtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    ccall((:cublasDtbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasCtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    ccall((:cublasCtbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasZtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    ccall((:cublasZtbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasStpmv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    ccall((:cublasStpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasDtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    ccall((:cublasDtpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasCtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    ccall((:cublasCtpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasZtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    ccall((:cublasZtpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasStrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    ccall((:cublasStrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasDtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    ccall((:cublasDtrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasCtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    ccall((:cublasCtrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasZtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    ccall((:cublasZtrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasStpsv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    ccall((:cublasStpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasDtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    ccall((:cublasDtpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasCtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    ccall((:cublasCtpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasZtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    ccall((:cublasZtpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasStbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    ccall((:cublasStbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasDtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    ccall((:cublasDtbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasCtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    ccall((:cublasCtbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasZtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    ccall((:cublasZtbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasSsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasSsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasDsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasCsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasCsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasZsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasChemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasChemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZhemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasZhemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasSsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasSsbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasDsbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasChbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasChbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZhbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    ccall((:cublasZhbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasSspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    ccall((:cublasSspmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasDspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    ccall((:cublasDspmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasChpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    ccall((:cublasChpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasZhpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    ccall((:cublasZhpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasSger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    ccall((:cublasSger_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasDger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    ccall((:cublasDger_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    ccall((:cublasCgeru_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    ccall((:cublasCgerc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    ccall((:cublasZgeru_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    ccall((:cublasZgerc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasSsyr_v2(handle, uplo, n, alpha, x, incx, A, lda)
    ccall((:cublasSsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasDsyr_v2(handle, uplo, n, alpha, x, incx, A, lda)
    ccall((:cublasDsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasCsyr_v2(handle, uplo, n, alpha, x, incx, A, lda)
    ccall((:cublasCsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasZsyr_v2(handle, uplo, n, alpha, x, incx, A, lda)
    ccall((:cublasZsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasCher_v2(handle, uplo, n, alpha, x, incx, A, lda)
    ccall((:cublasCher_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasZher_v2(handle, uplo, n, alpha, x, incx, A, lda)
    ccall((:cublasZher_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasSspr_v2(handle, uplo, n, alpha, x, incx, AP)
    ccall((:cublasSspr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, uplo, n, alpha, x, incx, AP)
end

function cublasDspr_v2(handle, uplo, n, alpha, x, incx, AP)
    ccall((:cublasDspr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, uplo, n, alpha, x, incx, AP)
end

function cublasChpr_v2(handle, uplo, n, alpha, x, incx, AP)
    ccall((:cublasChpr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint, Ptr{cuComplex}), handle, uplo, n, alpha, x, incx, AP)
end

function cublasZhpr_v2(handle, uplo, n, alpha, x, incx, AP)
    ccall((:cublasZhpr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}), handle, uplo, n, alpha, x, incx, AP)
end

function cublasSsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    ccall((:cublasSsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasDsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    ccall((:cublasDsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    ccall((:cublasCsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    ccall((:cublasZsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    ccall((:cublasCher2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    ccall((:cublasZher2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasSspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    ccall((:cublasSspr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasDspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    ccall((:cublasDspr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasChpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    ccall((:cublasChpr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}), handle, uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasZhpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    ccall((:cublasZhpr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}), handle, uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasSgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasDgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasCgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasCgemm3m, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCgemm3mEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    ccall((:cublasCgemm3mEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{Void}, cudaDataType, Cint, Ptr{Void}, cudaDataType, Cint, Ptr{cuComplex}, Ptr{Void}, cudaDataType, Cint), handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
end

function cublasZgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasZgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasZgemm3m, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    ccall((:cublasSgemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Void}, cudaDataType, Cint, Ptr{Void}, cudaDataType, Cint, Ptr{Cfloat}, Ptr{Void}, cudaDataType, Cint), handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
end

function cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)
    ccall((:cublasGemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Void}, Ptr{Void}, cudaDataType, Cint, Ptr{Void}, cudaDataType, Cint, Ptr{Void}, Ptr{Void}, cudaDataType, Cint, cudaDataType, cublasGemmAlgo_t), handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)
end

function cublasCgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    ccall((:cublasCgemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{Void}, cudaDataType, Cint, Ptr{Void}, cudaDataType, Cint, Ptr{cuComplex}, Ptr{Void}, cudaDataType, Cint), handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
end

function cublasUint8gemmBias(handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult, C_shift)
    ccall((:cublasUint8gemmBias, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cuchar}, Cint, Cint, Ptr{Cuchar}, Cint, Cint, Ptr{Cuchar}, Cint, Cint, Cint, Cint), handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult, C_shift)
end

function cublasSsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    ccall((:cublasSsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasDsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    ccall((:cublasDsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasCsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    ccall((:cublasCsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasZsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    ccall((:cublasZsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasCsyrkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    ccall((:cublasCsyrkEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{Void}, cudaDataType, Cint, Ptr{cuComplex}, Ptr{Void}, cudaDataType, Cint), handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
end

function cublasCsyrk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    ccall((:cublasCsyrk3mEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{Void}, cudaDataType, Cint, Ptr{cuComplex}, Ptr{Void}, cudaDataType, Cint), handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
end

function cublasCherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    ccall((:cublasCherk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasZherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    ccall((:cublasZherk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasCherkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    ccall((:cublasCherkEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Void}, cudaDataType, Cint, Ptr{Cfloat}, Ptr{Void}, cudaDataType, Cint), handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
end

function cublasCherk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    ccall((:cublasCherk3mEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Void}, cudaDataType, Cint, Ptr{Cfloat}, Ptr{Void}, cudaDataType, Cint), handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
end

function cublasSsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasSsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasDsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasCsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasZsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasCher2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasZher2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasSsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasSsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasDsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasCsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasZsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasCherkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasZherkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasSsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasSsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasDsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasCsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasZsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasChemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasChemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZhemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:cublasZhemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    ccall((:cublasStrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    ccall((:cublasDtrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasCtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    ccall((:cublasCtrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasZtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    ccall((:cublasZtrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasStrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    ccall((:cublasStrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    ccall((:cublasDtrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasCtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    ccall((:cublasCtrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasZtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    ccall((:cublasZtrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    ccall((:cublasSgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cfloat}, Ptr{Ptr{Cfloat}}, Cint, Cint), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end

function cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    ccall((:cublasDgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cdouble}, Ptr{Ptr{Cdouble}}, Cint, Cint), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end

function cublasCgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    ccall((:cublasCgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{cuComplex}, Ptr{Ptr{cuComplex}}, Cint, Cint), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end

function cublasCgemm3mBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    ccall((:cublasCgemm3mBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{cuComplex}, Ptr{Ptr{cuComplex}}, Cint, Cint), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end

function cublasZgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    ccall((:cublasZgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{cuDoubleComplex}, Ptr{Ptr{cuDoubleComplex}}, Cint, Cint), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end

function cublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo)
    ccall((:cublasGemmBatchedEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Void}, Ptr{Ptr{Void}}, cudaDataType, Cint, Ptr{Ptr{Void}}, cudaDataType, Cint, Ptr{Void}, Ptr{Ptr{Void}}, cudaDataType, Cint, Cint, cudaDataType, cublasGemmAlgo_t), handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo)
end

function cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo)
    ccall((:cublasGemmStridedBatchedEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Void}, Ptr{Void}, cudaDataType, Cint, Clonglong, Ptr{Void}, cudaDataType, Cint, Clonglong, Ptr{Void}, Ptr{Void}, cudaDataType, Cint, Clonglong, Cint, cudaDataType, cublasGemmAlgo_t), handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo)
end

function cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    ccall((:cublasSgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Clonglong, Ptr{Cfloat}, Cint, Clonglong, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Clonglong, Cint), handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
end

function cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    ccall((:cublasDgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Clonglong, Ptr{Cdouble}, Cint, Clonglong, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Clonglong, Cint), handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
end

function cublasCgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    ccall((:cublasCgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Clonglong, Ptr{cuComplex}, Cint, Clonglong, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Clonglong, Cint), handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
end

function cublasCgemm3mStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    ccall((:cublasCgemm3mStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Clonglong, Ptr{cuComplex}, Cint, Clonglong, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Clonglong, Cint), handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
end

function cublasZgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    ccall((:cublasZgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Clonglong, Ptr{cuDoubleComplex}, Cint, Clonglong, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Clonglong, Cint), handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
end

function cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    ccall((:cublasSgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end

function cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    ccall((:cublasDgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end

function cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    ccall((:cublasCgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end

function cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    ccall((:cublasZgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end

function cublasSgetrfBatched(handle, n, A, lda, P, info, batchSize)
    ccall((:cublasSgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, n, A, lda, P, info, batchSize)
end

function cublasDgetrfBatched(handle, n, A, lda, P, info, batchSize)
    ccall((:cublasDgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, n, A, lda, P, info, batchSize)
end

function cublasCgetrfBatched(handle, n, A, lda, P, info, batchSize)
    ccall((:cublasCgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, n, A, lda, P, info, batchSize)
end

function cublasZgetrfBatched(handle, n, A, lda, P, info, batchSize)
    ccall((:cublasZgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, n, A, lda, P, info, batchSize)
end

function cublasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)
    ccall((:cublasSgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, P, C, ldc, info, batchSize)
end

function cublasDgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)
    ccall((:cublasDgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, P, C, ldc, info, batchSize)
end

function cublasCgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)
    ccall((:cublasCgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, P, C, ldc, info, batchSize)
end

function cublasZgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)
    ccall((:cublasZgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, P, C, ldc, info, batchSize)
end

function cublasSgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
    ccall((:cublasSgetrsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Cint), handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
end

function cublasDgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
    ccall((:cublasDgetrsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Cint), handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
end

function cublasCgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
    ccall((:cublasCgetrsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Cint), handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
end

function cublasZgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
    ccall((:cublasZgetrsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Cint), handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
end

function cublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    ccall((:cublasStrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Cint, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end

function cublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    ccall((:cublasDtrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Cint, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end

function cublasCtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    ccall((:cublasCtrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end

function cublasZtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    ccall((:cublasZtrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end

function cublasSmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    ccall((:cublasSmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end

function cublasDmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    ccall((:cublasDmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end

function cublasCmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    ccall((:cublasCmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end

function cublasZmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    ccall((:cublasZmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end

function cublasSgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    ccall((:cublasSgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Ptr{Cint}, Cint), handle, m, n, Aarray, lda, TauArray, info, batchSize)
end

function cublasDgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    ccall((:cublasDgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Ptr{Cint}, Cint), handle, m, n, Aarray, lda, TauArray, info, batchSize)
end

function cublasCgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    ccall((:cublasCgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Ptr{Cint}, Cint), handle, m, n, Aarray, lda, TauArray, info, batchSize)
end

function cublasZgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    ccall((:cublasZgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Ptr{Cint}, Cint), handle, m, n, Aarray, lda, TauArray, info, batchSize)
end

function cublasSgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
    ccall((:cublasSgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
end

function cublasDgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
    ccall((:cublasDgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
end

function cublasCgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
    ccall((:cublasCgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
end

function cublasZgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
    ccall((:cublasZgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
end

function cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)
    ccall((:cublasSdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, mode, m, n, A, lda, x, incx, C, ldc)
end

function cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)
    ccall((:cublasDdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, mode, m, n, A, lda, x, incx, C, ldc)
end

function cublasCdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)
    ccall((:cublasCdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, mode, m, n, A, lda, x, incx, C, ldc)
end

function cublasZdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)
    ccall((:cublasZdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, mode, m, n, A, lda, x, incx, C, ldc)
end

function cublasStpttr(handle, uplo, n, AP, A, lda)
    ccall((:cublasStpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, n, AP, A, lda)
end

function cublasDtpttr(handle, uplo, n, AP, A, lda)
    ccall((:cublasDtpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, n, AP, A, lda)
end

function cublasCtpttr(handle, uplo, n, AP, A, lda)
    ccall((:cublasCtpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, n, AP, A, lda)
end

function cublasZtpttr(handle, uplo, n, AP, A, lda)
    ccall((:cublasZtpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, AP, A, lda)
end

function cublasStrttp(handle, uplo, n, A, lda, AP)
    ccall((:cublasStrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, uplo, n, A, lda, AP)
end

function cublasDtrttp(handle, uplo, n, A, lda, AP)
    ccall((:cublasDtrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, uplo, n, A, lda, AP)
end

function cublasCtrttp(handle, uplo, n, A, lda, AP)
    ccall((:cublasCtrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}), handle, uplo, n, A, lda, AP)
end

function cublasZtrttp(handle, uplo, n, A, lda, AP)
    ccall((:cublasZtrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}), handle, uplo, n, A, lda, AP)
end
