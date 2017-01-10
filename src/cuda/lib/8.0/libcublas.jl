# Julia wrapper for header: /home/shindo/local-lemon/cuda-8.0/include/cublas_v2.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function cublasCreate_v2(handle)
    check_cublasstatus(ccall((:cublasCreate_v2,libcublas),cublasStatus_t,(Ptr{cublasHandle_t},),handle))
end

function cublasDestroy_v2(handle)
    check_cublasstatus(ccall((:cublasDestroy_v2,libcublas),cublasStatus_t,(cublasHandle_t,),handle))
end

function cublasGetVersion_v2(handle,version)
    check_cublasstatus(ccall((:cublasGetVersion_v2,libcublas),cublasStatus_t,(cublasHandle_t,Ptr{Cint}),handle,version))
end

function cublasGetProperty(_type,value)
    check_cublasstatus(ccall((:cublasGetProperty,libcublas),cublasStatus_t,(libraryPropertyType,Ptr{Cint}),_type,value))
end

function cublasSetStream_v2(handle,streamId)
    check_cublasstatus(ccall((:cublasSetStream_v2,libcublas),cublasStatus_t,(cublasHandle_t,cudaStream_t),handle,streamId))
end

function cublasGetStream_v2(handle,streamId)
    check_cublasstatus(ccall((:cublasGetStream_v2,libcublas),cublasStatus_t,(cublasHandle_t,Ptr{cudaStream_t}),handle,streamId))
end

function cublasGetPointerMode_v2(handle,mode)
    check_cublasstatus(ccall((:cublasGetPointerMode_v2,libcublas),cublasStatus_t,(cublasHandle_t,Ptr{cublasPointerMode_t}),handle,mode))
end

function cublasSetPointerMode_v2(handle,mode)
    check_cublasstatus(ccall((:cublasSetPointerMode_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasPointerMode_t),handle,mode))
end

function cublasGetAtomicsMode(handle,mode)
    check_cublasstatus(ccall((:cublasGetAtomicsMode,libcublas),cublasStatus_t,(cublasHandle_t,Ptr{cublasAtomicsMode_t}),handle,mode))
end

function cublasSetAtomicsMode(handle,mode)
    check_cublasstatus(ccall((:cublasSetAtomicsMode,libcublas),cublasStatus_t,(cublasHandle_t,cublasAtomicsMode_t),handle,mode))
end

function cublasSetVector(n,elemSize,x,incx,devicePtr,incy)
    check_cublasstatus(ccall((:cublasSetVector,libcublas),cublasStatus_t,(Cint,Cint,Ptr{Void},Cint,Ptr{Void},Cint),n,elemSize,x,incx,devicePtr,incy))
end

function cublasGetVector(n,elemSize,x,incx,y,incy)
    check_cublasstatus(ccall((:cublasGetVector,libcublas),cublasStatus_t,(Cint,Cint,Ptr{Void},Cint,Ptr{Void},Cint),n,elemSize,x,incx,y,incy))
end

function cublasSetMatrix(rows,cols,elemSize,A,lda,B,ldb)
    check_cublasstatus(ccall((:cublasSetMatrix,libcublas),cublasStatus_t,(Cint,Cint,Cint,Ptr{Void},Cint,Ptr{Void},Cint),rows,cols,elemSize,A,lda,B,ldb))
end

function cublasGetMatrix(rows,cols,elemSize,A,lda,B,ldb)
    check_cublasstatus(ccall((:cublasGetMatrix,libcublas),cublasStatus_t,(Cint,Cint,Cint,Ptr{Void},Cint,Ptr{Void},Cint),rows,cols,elemSize,A,lda,B,ldb))
end

function cublasSetVectorAsync(n,elemSize,hostPtr,incx,devicePtr,incy,stream)
    check_cublasstatus(ccall((:cublasSetVectorAsync,libcublas),cublasStatus_t,(Cint,Cint,Ptr{Void},Cint,Ptr{Void},Cint,cudaStream_t),n,elemSize,hostPtr,incx,devicePtr,incy,stream))
end

function cublasGetVectorAsync(n,elemSize,devicePtr,incx,hostPtr,incy,stream)
    check_cublasstatus(ccall((:cublasGetVectorAsync,libcublas),cublasStatus_t,(Cint,Cint,Ptr{Void},Cint,Ptr{Void},Cint,cudaStream_t),n,elemSize,devicePtr,incx,hostPtr,incy,stream))
end

function cublasSetMatrixAsync(rows,cols,elemSize,A,lda,B,ldb,stream)
    check_cublasstatus(ccall((:cublasSetMatrixAsync,libcublas),cublasStatus_t,(Cint,Cint,Cint,Ptr{Void},Cint,Ptr{Void},Cint,cudaStream_t),rows,cols,elemSize,A,lda,B,ldb,stream))
end

function cublasGetMatrixAsync(rows,cols,elemSize,A,lda,B,ldb,stream)
    check_cublasstatus(ccall((:cublasGetMatrixAsync,libcublas),cublasStatus_t,(Cint,Cint,Cint,Ptr{Void},Cint,Ptr{Void},Cint,cudaStream_t),rows,cols,elemSize,A,lda,B,ldb,stream))
end

function cublasXerbla(srName,info)
    ccall((:cublasXerbla,libcublas),Void,(Ptr{UInt8},Cint),srName,info)
end

function cublasNrm2Ex(handle,n,x,xType,incx,result,resultType,executionType)
    check_cublasstatus(ccall((:cublasNrm2Ex,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Void},cudaDataType,Cint,Ptr{Void},cudaDataType,cudaDataType),handle,n,x,xType,incx,result,resultType,executionType))
end

function cublasSnrm2_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasSnrm2_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat}),handle,n,x,incx,result))
end

function cublasDnrm2_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasDnrm2_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble}),handle,n,x,incx,result))
end

function cublasScnrm2_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasScnrm2_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuComplex},Cint,Ptr{Cfloat}),handle,n,x,incx,result))
end

function cublasDznrm2_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasDznrm2_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuDoubleComplex},Cint,Ptr{Cdouble}),handle,n,x,incx,result))
end

function cublasDotEx(handle,n,x,xType,incx,y,yType,incy,result,resultType,executionType)
    check_cublasstatus(ccall((:cublasDotEx,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Void},cudaDataType,Cint,Ptr{Void},cudaDataType,Cint,Ptr{Void},cudaDataType,cudaDataType),handle,n,x,xType,incx,y,yType,incy,result,resultType,executionType))
end

function cublasDotcEx(handle,n,x,xType,incx,y,yType,incy,result,resultType,executionType)
    check_cublasstatus(ccall((:cublasDotcEx,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Void},cudaDataType,Cint,Ptr{Void},cudaDataType,Cint,Ptr{Void},cudaDataType,cudaDataType),handle,n,x,xType,incx,y,yType,incy,result,resultType,executionType))
end

function cublasSdot_v2(handle,n,x,incx,y,incy,result)
    check_cublasstatus(ccall((:cublasSdot_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat}),handle,n,x,incx,y,incy,result))
end

function cublasDdot_v2(handle,n,x,incx,y,incy,result)
    check_cublasstatus(ccall((:cublasDdot_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble}),handle,n,x,incx,y,incy,result))
end

function cublasCdotu_v2(handle,n,x,incx,y,incy,result)
    check_cublasstatus(ccall((:cublasCdotu_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex}),handle,n,x,incx,y,incy,result))
end

function cublasCdotc_v2(handle,n,x,incx,y,incy,result)
    check_cublasstatus(ccall((:cublasCdotc_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex}),handle,n,x,incx,y,incy,result))
end

function cublasZdotu_v2(handle,n,x,incx,y,incy,result)
    check_cublasstatus(ccall((:cublasZdotu_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex}),handle,n,x,incx,y,incy,result))
end

function cublasZdotc_v2(handle,n,x,incx,y,incy,result)
    check_cublasstatus(ccall((:cublasZdotc_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex}),handle,n,x,incx,y,incy,result))
end

function cublasScalEx(handle,n,alpha,alphaType,x,xType,incx,executionType)
    check_cublasstatus(ccall((:cublasScalEx,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Void},cudaDataType,Ptr{Void},cudaDataType,Cint,cudaDataType),handle,n,alpha,alphaType,x,xType,incx,executionType))
end

function cublasSscal_v2(handle,n,alpha,x,incx)
    check_cublasstatus(ccall((:cublasSscal_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,n,alpha,x,incx))
end

function cublasDscal_v2(handle,n,alpha,x,incx)
    check_cublasstatus(ccall((:cublasDscal_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,n,alpha,x,incx))
end

function cublasCscal_v2(handle,n,alpha,x,incx)
    check_cublasstatus(ccall((:cublasCscal_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,n,alpha,x,incx))
end

function cublasCsscal_v2(handle,n,alpha,x,incx)
    check_cublasstatus(ccall((:cublasCsscal_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cfloat},Ptr{cuComplex},Cint),handle,n,alpha,x,incx))
end

function cublasZscal_v2(handle,n,alpha,x,incx)
    check_cublasstatus(ccall((:cublasZscal_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,n,alpha,x,incx))
end

function cublasZdscal_v2(handle,n,alpha,x,incx)
    check_cublasstatus(ccall((:cublasZdscal_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cdouble},Ptr{cuDoubleComplex},Cint),handle,n,alpha,x,incx))
end

function cublasAxpyEx(handle,n,alpha,alphaType,x,xType,incx,y,yType,incy,executiontype)
    check_cublasstatus(ccall((:cublasAxpyEx,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Void},cudaDataType,Ptr{Void},cudaDataType,Cint,Ptr{Void},cudaDataType,Cint,cudaDataType),handle,n,alpha,alphaType,x,xType,incx,y,yType,incy,executiontype))
end

function cublasSaxpy_v2(handle,n,alpha,x,incx,y,incy)
    check_cublasstatus(ccall((:cublasSaxpy_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,n,alpha,x,incx,y,incy))
end

function cublasDaxpy_v2(handle,n,alpha,x,incx,y,incy)
    check_cublasstatus(ccall((:cublasDaxpy_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,n,alpha,x,incx,y,incy))
end

function cublasCaxpy_v2(handle,n,alpha,x,incx,y,incy)
    check_cublasstatus(ccall((:cublasCaxpy_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,n,alpha,x,incx,y,incy))
end

function cublasZaxpy_v2(handle,n,alpha,x,incx,y,incy)
    check_cublasstatus(ccall((:cublasZaxpy_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,n,alpha,x,incx,y,incy))
end

function cublasScopy_v2(handle,n,x,incx,y,incy)
    check_cublasstatus(ccall((:cublasScopy_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,n,x,incx,y,incy))
end

function cublasDcopy_v2(handle,n,x,incx,y,incy)
    check_cublasstatus(ccall((:cublasDcopy_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,n,x,incx,y,incy))
end

function cublasCcopy_v2(handle,n,x,incx,y,incy)
    check_cublasstatus(ccall((:cublasCcopy_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,n,x,incx,y,incy))
end

function cublasZcopy_v2(handle,n,x,incx,y,incy)
    check_cublasstatus(ccall((:cublasZcopy_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,n,x,incx,y,incy))
end

function cublasSswap_v2(handle,n,x,incx,y,incy)
    check_cublasstatus(ccall((:cublasSswap_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,n,x,incx,y,incy))
end

function cublasDswap_v2(handle,n,x,incx,y,incy)
    check_cublasstatus(ccall((:cublasDswap_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,n,x,incx,y,incy))
end

function cublasCswap_v2(handle,n,x,incx,y,incy)
    check_cublasstatus(ccall((:cublasCswap_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,n,x,incx,y,incy))
end

function cublasZswap_v2(handle,n,x,incx,y,incy)
    check_cublasstatus(ccall((:cublasZswap_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,n,x,incx,y,incy))
end

function cublasIsamax_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasIsamax_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cfloat},Cint,Ptr{Cint}),handle,n,x,incx,result))
end

function cublasIdamax_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasIdamax_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cdouble},Cint,Ptr{Cint}),handle,n,x,incx,result))
end

function cublasIcamax_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasIcamax_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuComplex},Cint,Ptr{Cint}),handle,n,x,incx,result))
end

function cublasIzamax_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasIzamax_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuDoubleComplex},Cint,Ptr{Cint}),handle,n,x,incx,result))
end

function cublasIsamin_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasIsamin_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cfloat},Cint,Ptr{Cint}),handle,n,x,incx,result))
end

function cublasIdamin_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasIdamin_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cdouble},Cint,Ptr{Cint}),handle,n,x,incx,result))
end

function cublasIcamin_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasIcamin_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuComplex},Cint,Ptr{Cint}),handle,n,x,incx,result))
end

function cublasIzamin_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasIzamin_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuDoubleComplex},Cint,Ptr{Cint}),handle,n,x,incx,result))
end

function cublasSasum_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasSasum_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat}),handle,n,x,incx,result))
end

function cublasDasum_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasDasum_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble}),handle,n,x,incx,result))
end

function cublasScasum_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasScasum_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuComplex},Cint,Ptr{Cfloat}),handle,n,x,incx,result))
end

function cublasDzasum_v2(handle,n,x,incx,result)
    check_cublasstatus(ccall((:cublasDzasum_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuDoubleComplex},Cint,Ptr{Cdouble}),handle,n,x,incx,result))
end

function cublasSrot_v2(handle,n,x,incx,y,incy,c,s)
    check_cublasstatus(ccall((:cublasSrot_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cfloat}),handle,n,x,incx,y,incy,c,s))
end

function cublasDrot_v2(handle,n,x,incx,y,incy,c,s)
    check_cublasstatus(ccall((:cublasDrot_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble}),handle,n,x,incx,y,incy,c,s))
end

function cublasCrot_v2(handle,n,x,incx,y,incy,c,s)
    check_cublasstatus(ccall((:cublasCrot_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{Cfloat},Ptr{cuComplex}),handle,n,x,incx,y,incy,c,s))
end

function cublasCsrot_v2(handle,n,x,incx,y,incy,c,s)
    check_cublasstatus(ccall((:cublasCsrot_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{Cfloat},Ptr{Cfloat}),handle,n,x,incx,y,incy,c,s))
end

function cublasZrot_v2(handle,n,x,incx,y,incy,c,s)
    check_cublasstatus(ccall((:cublasZrot_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{Cdouble},Ptr{cuDoubleComplex}),handle,n,x,incx,y,incy,c,s))
end

function cublasZdrot_v2(handle,n,x,incx,y,incy,c,s)
    check_cublasstatus(ccall((:cublasZdrot_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{Cdouble},Ptr{Cdouble}),handle,n,x,incx,y,incy,c,s))
end

function cublasSrotg_v2(handle,a,b,c,s)
    check_cublasstatus(ccall((:cublasSrotg_v2,libcublas),cublasStatus_t,(cublasHandle_t,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),handle,a,b,c,s))
end

function cublasDrotg_v2(handle,a,b,c,s)
    check_cublasstatus(ccall((:cublasDrotg_v2,libcublas),cublasStatus_t,(cublasHandle_t,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),handle,a,b,c,s))
end

function cublasCrotg_v2(handle,a,b,c,s)
    check_cublasstatus(ccall((:cublasCrotg_v2,libcublas),cublasStatus_t,(cublasHandle_t,Ptr{cuComplex},Ptr{cuComplex},Ptr{Cfloat},Ptr{cuComplex}),handle,a,b,c,s))
end

function cublasZrotg_v2(handle,a,b,c,s)
    check_cublasstatus(ccall((:cublasZrotg_v2,libcublas),cublasStatus_t,(cublasHandle_t,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Ptr{Cdouble},Ptr{cuDoubleComplex}),handle,a,b,c,s))
end

function cublasSrotm_v2(handle,n,x,incx,y,incy,param)
    check_cublasstatus(ccall((:cublasSrotm_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat}),handle,n,x,incx,y,incy,param))
end

function cublasDrotm_v2(handle,n,x,incx,y,incy,param)
    check_cublasstatus(ccall((:cublasDrotm_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble}),handle,n,x,incx,y,incy,param))
end

function cublasSrotmg_v2(handle,d1,d2,x1,y1,param)
    check_cublasstatus(ccall((:cublasSrotmg_v2,libcublas),cublasStatus_t,(cublasHandle_t,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),handle,d1,d2,x1,y1,param))
end

function cublasDrotmg_v2(handle,d1,d2,x1,y1,param)
    check_cublasstatus(ccall((:cublasDrotmg_v2,libcublas),cublasStatus_t,(cublasHandle_t,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),handle,d1,d2,x1,y1,param))
end

function cublasSgemv_v2(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasSgemv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasDgemv_v2(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasDgemv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasCgemv_v2(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasCgemv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasZgemv_v2(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasZgemv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasSgbmv_v2(handle,trans,m,n,kl,ku,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasSgbmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,trans,m,n,kl,ku,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasDgbmv_v2(handle,trans,m,n,kl,ku,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasDgbmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,trans,m,n,kl,ku,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasCgbmv_v2(handle,trans,m,n,kl,ku,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasCgbmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,trans,m,n,kl,ku,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasZgbmv_v2(handle,trans,m,n,kl,ku,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasZgbmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,trans,m,n,kl,ku,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasStrmv_v2(handle,uplo,trans,diag,n,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasStrmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,uplo,trans,diag,n,A,lda,x,incx))
end

function cublasDtrmv_v2(handle,uplo,trans,diag,n,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasDtrmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,uplo,trans,diag,n,A,lda,x,incx))
end

function cublasCtrmv_v2(handle,uplo,trans,diag,n,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasCtrmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,uplo,trans,diag,n,A,lda,x,incx))
end

function cublasZtrmv_v2(handle,uplo,trans,diag,n,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasZtrmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,uplo,trans,diag,n,A,lda,x,incx))
end

function cublasStbmv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasStbmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,uplo,trans,diag,n,k,A,lda,x,incx))
end

function cublasDtbmv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasDtbmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,uplo,trans,diag,n,k,A,lda,x,incx))
end

function cublasCtbmv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasCtbmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,uplo,trans,diag,n,k,A,lda,x,incx))
end

function cublasZtbmv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasZtbmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,uplo,trans,diag,n,k,A,lda,x,incx))
end

function cublasStpmv_v2(handle,uplo,trans,diag,n,AP,x,incx)
    check_cublasstatus(ccall((:cublasStpmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,uplo,trans,diag,n,AP,x,incx))
end

function cublasDtpmv_v2(handle,uplo,trans,diag,n,AP,x,incx)
    check_cublasstatus(ccall((:cublasDtpmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,uplo,trans,diag,n,AP,x,incx))
end

function cublasCtpmv_v2(handle,uplo,trans,diag,n,AP,x,incx)
    check_cublasstatus(ccall((:cublasCtpmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,uplo,trans,diag,n,AP,x,incx))
end

function cublasZtpmv_v2(handle,uplo,trans,diag,n,AP,x,incx)
    check_cublasstatus(ccall((:cublasZtpmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,uplo,trans,diag,n,AP,x,incx))
end

function cublasStrsv_v2(handle,uplo,trans,diag,n,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasStrsv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,uplo,trans,diag,n,A,lda,x,incx))
end

function cublasDtrsv_v2(handle,uplo,trans,diag,n,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasDtrsv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,uplo,trans,diag,n,A,lda,x,incx))
end

function cublasCtrsv_v2(handle,uplo,trans,diag,n,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasCtrsv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,uplo,trans,diag,n,A,lda,x,incx))
end

function cublasZtrsv_v2(handle,uplo,trans,diag,n,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasZtrsv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,uplo,trans,diag,n,A,lda,x,incx))
end

function cublasStpsv_v2(handle,uplo,trans,diag,n,AP,x,incx)
    check_cublasstatus(ccall((:cublasStpsv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,uplo,trans,diag,n,AP,x,incx))
end

function cublasDtpsv_v2(handle,uplo,trans,diag,n,AP,x,incx)
    check_cublasstatus(ccall((:cublasDtpsv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,uplo,trans,diag,n,AP,x,incx))
end

function cublasCtpsv_v2(handle,uplo,trans,diag,n,AP,x,incx)
    check_cublasstatus(ccall((:cublasCtpsv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,uplo,trans,diag,n,AP,x,incx))
end

function cublasZtpsv_v2(handle,uplo,trans,diag,n,AP,x,incx)
    check_cublasstatus(ccall((:cublasZtpsv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,uplo,trans,diag,n,AP,x,incx))
end

function cublasStbsv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasStbsv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,uplo,trans,diag,n,k,A,lda,x,incx))
end

function cublasDtbsv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasDtbsv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,uplo,trans,diag,n,k,A,lda,x,incx))
end

function cublasCtbsv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasCtbsv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,uplo,trans,diag,n,k,A,lda,x,incx))
end

function cublasZtbsv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx)
    check_cublasstatus(ccall((:cublasZtbsv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,uplo,trans,diag,n,k,A,lda,x,incx))
end

function cublasSsymv_v2(handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasSsymv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasDsymv_v2(handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasDsymv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasCsymv_v2(handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasCsymv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasZsymv_v2(handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasZsymv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasChemv_v2(handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasChemv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasZhemv_v2(handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasZhemv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasSsbmv_v2(handle,uplo,n,k,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasSsbmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,uplo,n,k,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasDsbmv_v2(handle,uplo,n,k,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasDsbmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,uplo,n,k,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasChbmv_v2(handle,uplo,n,k,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasChbmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,uplo,n,k,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasZhbmv_v2(handle,uplo,n,k,alpha,A,lda,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasZhbmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,uplo,n,k,alpha,A,lda,x,incx,beta,y,incy))
end

function cublasSspmv_v2(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasSspmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,uplo,n,alpha,AP,x,incx,beta,y,incy))
end

function cublasDspmv_v2(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasDspmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,uplo,n,alpha,AP,x,incx,beta,y,incy))
end

function cublasChpmv_v2(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasChpmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuComplex},Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,uplo,n,alpha,AP,x,incx,beta,y,incy))
end

function cublasZhpmv_v2(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)
    check_cublasstatus(ccall((:cublasZhpmv_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,uplo,n,alpha,AP,x,incx,beta,y,incy))
end

function cublasSger_v2(handle,m,n,alpha,x,incx,y,incy,A,lda)
    check_cublasstatus(ccall((:cublasSger_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,m,n,alpha,x,incx,y,incy,A,lda))
end

function cublasDger_v2(handle,m,n,alpha,x,incx,y,incy,A,lda)
    check_cublasstatus(ccall((:cublasDger_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,m,n,alpha,x,incx,y,incy,A,lda))
end

function cublasCgeru_v2(handle,m,n,alpha,x,incx,y,incy,A,lda)
    check_cublasstatus(ccall((:cublasCgeru_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,m,n,alpha,x,incx,y,incy,A,lda))
end

function cublasCgerc_v2(handle,m,n,alpha,x,incx,y,incy,A,lda)
    check_cublasstatus(ccall((:cublasCgerc_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,m,n,alpha,x,incx,y,incy,A,lda))
end

function cublasZgeru_v2(handle,m,n,alpha,x,incx,y,incy,A,lda)
    check_cublasstatus(ccall((:cublasZgeru_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,m,n,alpha,x,incx,y,incy,A,lda))
end

function cublasZgerc_v2(handle,m,n,alpha,x,incx,y,incy,A,lda)
    check_cublasstatus(ccall((:cublasZgerc_v2,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,m,n,alpha,x,incx,y,incy,A,lda))
end

function cublasSsyr_v2(handle,uplo,n,alpha,x,incx,A,lda)
    check_cublasstatus(ccall((:cublasSsyr_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,uplo,n,alpha,x,incx,A,lda))
end

function cublasDsyr_v2(handle,uplo,n,alpha,x,incx,A,lda)
    check_cublasstatus(ccall((:cublasDsyr_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,uplo,n,alpha,x,incx,A,lda))
end

function cublasCsyr_v2(handle,uplo,n,alpha,x,incx,A,lda)
    check_cublasstatus(ccall((:cublasCsyr_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,uplo,n,alpha,x,incx,A,lda))
end

function cublasZsyr_v2(handle,uplo,n,alpha,x,incx,A,lda)
    check_cublasstatus(ccall((:cublasZsyr_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,uplo,n,alpha,x,incx,A,lda))
end

function cublasCher_v2(handle,uplo,n,alpha,x,incx,A,lda)
    check_cublasstatus(ccall((:cublasCher_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cfloat},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,uplo,n,alpha,x,incx,A,lda))
end

function cublasZher_v2(handle,uplo,n,alpha,x,incx,A,lda)
    check_cublasstatus(ccall((:cublasZher_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cdouble},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,uplo,n,alpha,x,incx,A,lda))
end

function cublasSspr_v2(handle,uplo,n,alpha,x,incx,AP)
    check_cublasstatus(ccall((:cublasSspr_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat}),handle,uplo,n,alpha,x,incx,AP))
end

function cublasDspr_v2(handle,uplo,n,alpha,x,incx,AP)
    check_cublasstatus(ccall((:cublasDspr_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble}),handle,uplo,n,alpha,x,incx,AP))
end

function cublasChpr_v2(handle,uplo,n,alpha,x,incx,AP)
    check_cublasstatus(ccall((:cublasChpr_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cfloat},Ptr{cuComplex},Cint,Ptr{cuComplex}),handle,uplo,n,alpha,x,incx,AP))
end

function cublasZhpr_v2(handle,uplo,n,alpha,x,incx,AP)
    check_cublasstatus(ccall((:cublasZhpr_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cdouble},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex}),handle,uplo,n,alpha,x,incx,AP))
end

function cublasSsyr2_v2(handle,uplo,n,alpha,x,incx,y,incy,A,lda)
    check_cublasstatus(ccall((:cublasSsyr2_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,uplo,n,alpha,x,incx,y,incy,A,lda))
end

function cublasDsyr2_v2(handle,uplo,n,alpha,x,incx,y,incy,A,lda)
    check_cublasstatus(ccall((:cublasDsyr2_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,uplo,n,alpha,x,incx,y,incy,A,lda))
end

function cublasCsyr2_v2(handle,uplo,n,alpha,x,incx,y,incy,A,lda)
    check_cublasstatus(ccall((:cublasCsyr2_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,uplo,n,alpha,x,incx,y,incy,A,lda))
end

function cublasZsyr2_v2(handle,uplo,n,alpha,x,incx,y,incy,A,lda)
    check_cublasstatus(ccall((:cublasZsyr2_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,uplo,n,alpha,x,incx,y,incy,A,lda))
end

function cublasCher2_v2(handle,uplo,n,alpha,x,incx,y,incy,A,lda)
    check_cublasstatus(ccall((:cublasCher2_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,uplo,n,alpha,x,incx,y,incy,A,lda))
end

function cublasZher2_v2(handle,uplo,n,alpha,x,incx,y,incy,A,lda)
    check_cublasstatus(ccall((:cublasZher2_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,uplo,n,alpha,x,incx,y,incy,A,lda))
end

function cublasSspr2_v2(handle,uplo,n,alpha,x,incx,y,incy,AP)
    check_cublasstatus(ccall((:cublasSspr2_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat}),handle,uplo,n,alpha,x,incx,y,incy,AP))
end

function cublasDspr2_v2(handle,uplo,n,alpha,x,incx,y,incy,AP)
    check_cublasstatus(ccall((:cublasDspr2_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble}),handle,uplo,n,alpha,x,incx,y,incy,AP))
end

function cublasChpr2_v2(handle,uplo,n,alpha,x,incx,y,incy,AP)
    check_cublasstatus(ccall((:cublasChpr2_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex}),handle,uplo,n,alpha,x,incx,y,incy,AP))
end

function cublasZhpr2_v2(handle,uplo,n,alpha,x,incx,y,incy,AP)
    check_cublasstatus(ccall((:cublasZhpr2_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex}),handle,uplo,n,alpha,x,incx,y,incy,AP))
end

function cublasSgemm_v2(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasSgemm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasDgemm_v2(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasDgemm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasCgemm_v2(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasCgemm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasCgemm3m(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasCgemm3m,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasCgemm3mEx(handle,transa,transb,m,n,k,alpha,A,Atype,lda,B,Btype,ldb,beta,C,Ctype,ldc)
    check_cublasstatus(ccall((:cublasCgemm3mEx,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{cuComplex},Ptr{Void},cudaDataType,Cint,Ptr{Void},cudaDataType,Cint,Ptr{cuComplex},Ptr{Void},cudaDataType,Cint),handle,transa,transb,m,n,k,alpha,A,Atype,lda,B,Btype,ldb,beta,C,Ctype,ldc))
end

function cublasZgemm_v2(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasZgemm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasZgemm3m(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasZgemm3m,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasHgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasHgemm,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{__half},Ptr{__half},Cint,Ptr{__half},Cint,Ptr{__half},Ptr{__half},Cint),handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasSgemmEx(handle,transa,transb,m,n,k,alpha,A,Atype,lda,B,Btype,ldb,beta,C,Ctype,ldc)
    check_cublasstatus(ccall((:cublasSgemmEx,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{Cfloat},Ptr{Void},cudaDataType,Cint,Ptr{Void},cudaDataType,Cint,Ptr{Cfloat},Ptr{Void},cudaDataType,Cint),handle,transa,transb,m,n,k,alpha,A,Atype,lda,B,Btype,ldb,beta,C,Ctype,ldc))
end

function cublasGemmEx(handle,transa,transb,m,n,k,alpha,A,Atype,lda,B,Btype,ldb,beta,C,Ctype,ldc,computeType,algo)
    check_cublasstatus(ccall((:cublasGemmEx,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{Void},Ptr{Void},cudaDataType,Cint,Ptr{Void},cudaDataType,Cint,Ptr{Void},Ptr{Void},cudaDataType,Cint,cudaDataType,cublasGemmAlgo_t),handle,transa,transb,m,n,k,alpha,A,Atype,lda,B,Btype,ldb,beta,C,Ctype,ldc,computeType,algo))
end

function cublasCgemmEx(handle,transa,transb,m,n,k,alpha,A,Atype,lda,B,Btype,ldb,beta,C,Ctype,ldc)
    check_cublasstatus(ccall((:cublasCgemmEx,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{cuComplex},Ptr{Void},cudaDataType,Cint,Ptr{Void},cudaDataType,Cint,Ptr{cuComplex},Ptr{Void},cudaDataType,Cint),handle,transa,transb,m,n,k,alpha,A,Atype,lda,B,Btype,ldb,beta,C,Ctype,ldc))
end

function cublasUint8gemmBias(handle,transa,transb,transc,m,n,k,A,A_bias,lda,B,B_bias,ldb,C,C_bias,ldc,C_mult,C_shift)
    check_cublasstatus(ccall((:cublasUint8gemmBias,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{Cuchar},Cint,Cint,Ptr{Cuchar},Cint,Cint,Ptr{Cuchar},Cint,Cint,Cint,Cint),handle,transa,transb,transc,m,n,k,A,A_bias,lda,B,B_bias,ldb,C,C_bias,ldc,C_mult,C_shift))
end

function cublasSsyrk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc)
    check_cublasstatus(ccall((:cublasSsyrk_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc))
end

function cublasDsyrk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc)
    check_cublasstatus(ccall((:cublasDsyrk_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc))
end

function cublasCsyrk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc)
    check_cublasstatus(ccall((:cublasCsyrk_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc))
end

function cublasZsyrk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc)
    check_cublasstatus(ccall((:cublasZsyrk_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc))
end

function cublasCsyrkEx(handle,uplo,trans,n,k,alpha,A,Atype,lda,beta,C,Ctype,ldc)
    check_cublasstatus(ccall((:cublasCsyrkEx,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{cuComplex},Ptr{Void},cudaDataType,Cint,Ptr{cuComplex},Ptr{Void},cudaDataType,Cint),handle,uplo,trans,n,k,alpha,A,Atype,lda,beta,C,Ctype,ldc))
end

function cublasCsyrk3mEx(handle,uplo,trans,n,k,alpha,A,Atype,lda,beta,C,Ctype,ldc)
    check_cublasstatus(ccall((:cublasCsyrk3mEx,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{cuComplex},Ptr{Void},cudaDataType,Cint,Ptr{cuComplex},Ptr{Void},cudaDataType,Cint),handle,uplo,trans,n,k,alpha,A,Atype,lda,beta,C,Ctype,ldc))
end

function cublasCherk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc)
    check_cublasstatus(ccall((:cublasCherk_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{Cfloat},Ptr{cuComplex},Cint,Ptr{Cfloat},Ptr{cuComplex},Cint),handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc))
end

function cublasZherk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc)
    check_cublasstatus(ccall((:cublasZherk_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{Cdouble},Ptr{cuDoubleComplex},Cint,Ptr{Cdouble},Ptr{cuDoubleComplex},Cint),handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc))
end

function cublasCherkEx(handle,uplo,trans,n,k,alpha,A,Atype,lda,beta,C,Ctype,ldc)
    check_cublasstatus(ccall((:cublasCherkEx,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{Cfloat},Ptr{Void},cudaDataType,Cint,Ptr{Cfloat},Ptr{Void},cudaDataType,Cint),handle,uplo,trans,n,k,alpha,A,Atype,lda,beta,C,Ctype,ldc))
end

function cublasCherk3mEx(handle,uplo,trans,n,k,alpha,A,Atype,lda,beta,C,Ctype,ldc)
    check_cublasstatus(ccall((:cublasCherk3mEx,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{Cfloat},Ptr{Void},cudaDataType,Cint,Ptr{Cfloat},Ptr{Void},cudaDataType,Cint),handle,uplo,trans,n,k,alpha,A,Atype,lda,beta,C,Ctype,ldc))
end

function cublasSsyr2k_v2(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasSsyr2k_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasDsyr2k_v2(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasDsyr2k_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasCsyr2k_v2(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasCsyr2k_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasZsyr2k_v2(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasZsyr2k_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasCher2k_v2(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasCher2k_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{Cfloat},Ptr{cuComplex},Cint),handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasZher2k_v2(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasZher2k_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{Cdouble},Ptr{cuDoubleComplex},Cint),handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasSsyrkx(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasSsyrkx,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasDsyrkx(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasDsyrkx,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasCsyrkx(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasCsyrkx,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasZsyrkx(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasZsyrkx,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasCherkx(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasCherkx,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{Cfloat},Ptr{cuComplex},Cint),handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasZherkx(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasZherkx,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,cublasOperation_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{Cdouble},Ptr{cuDoubleComplex},Cint),handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasSsymm_v2(handle,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasSsymm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasDsymm_v2(handle,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasDsymm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasCsymm_v2(handle,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasCsymm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasZsymm_v2(handle,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasZsymm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasChemm_v2(handle,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasChemm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasZhemm_v2(handle,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc)
    check_cublasstatus(ccall((:cublasZhemm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc))
end

function cublasStrsm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb)
    check_cublasstatus(ccall((:cublasStrsm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb))
end

function cublasDtrsm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb)
    check_cublasstatus(ccall((:cublasDtrsm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb))
end

function cublasCtrsm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb)
    check_cublasstatus(ccall((:cublasCtrsm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb))
end

function cublasZtrsm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb)
    check_cublasstatus(ccall((:cublasZtrsm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb))
end

function cublasStrmm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,C,ldc)
    check_cublasstatus(ccall((:cublasStrmm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,C,ldc))
end

function cublasDtrmm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,C,ldc)
    check_cublasstatus(ccall((:cublasDtrmm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,C,ldc))
end

function cublasCtrmm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,C,ldc)
    check_cublasstatus(ccall((:cublasCtrmm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,C,ldc))
end

function cublasZtrmm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,C,ldc)
    check_cublasstatus(ccall((:cublasZtrmm_v2,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,C,ldc))
end

function cublasSgemmBatched(handle,transa,transb,m,n,k,alpha,Aarray,lda,Barray,ldb,beta,Carray,ldc,batchCount)
    check_cublasstatus(ccall((:cublasSgemmBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{Cfloat},Ptr{Ptr{Cfloat}},Cint,Ptr{Ptr{Cfloat}},Cint,Ptr{Cfloat},Ptr{Ptr{Cfloat}},Cint,Cint),handle,transa,transb,m,n,k,alpha,Aarray,lda,Barray,ldb,beta,Carray,ldc,batchCount))
end

function cublasDgemmBatched(handle,transa,transb,m,n,k,alpha,Aarray,lda,Barray,ldb,beta,Carray,ldc,batchCount)
    check_cublasstatus(ccall((:cublasDgemmBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{Cdouble},Ptr{Ptr{Cdouble}},Cint,Ptr{Ptr{Cdouble}},Cint,Ptr{Cdouble},Ptr{Ptr{Cdouble}},Cint,Cint),handle,transa,transb,m,n,k,alpha,Aarray,lda,Barray,ldb,beta,Carray,ldc,batchCount))
end

function cublasCgemmBatched(handle,transa,transb,m,n,k,alpha,Aarray,lda,Barray,ldb,beta,Carray,ldc,batchCount)
    check_cublasstatus(ccall((:cublasCgemmBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{cuComplex},Ptr{Ptr{cuComplex}},Cint,Ptr{Ptr{cuComplex}},Cint,Ptr{cuComplex},Ptr{Ptr{cuComplex}},Cint,Cint),handle,transa,transb,m,n,k,alpha,Aarray,lda,Barray,ldb,beta,Carray,ldc,batchCount))
end

function cublasCgemm3mBatched(handle,transa,transb,m,n,k,alpha,Aarray,lda,Barray,ldb,beta,Carray,ldc,batchCount)
    check_cublasstatus(ccall((:cublasCgemm3mBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{cuComplex},Ptr{Ptr{cuComplex}},Cint,Ptr{Ptr{cuComplex}},Cint,Ptr{cuComplex},Ptr{Ptr{cuComplex}},Cint,Cint),handle,transa,transb,m,n,k,alpha,Aarray,lda,Barray,ldb,beta,Carray,ldc,batchCount))
end

function cublasZgemmBatched(handle,transa,transb,m,n,k,alpha,Aarray,lda,Barray,ldb,beta,Carray,ldc,batchCount)
    check_cublasstatus(ccall((:cublasZgemmBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{cuDoubleComplex},Ptr{Ptr{cuDoubleComplex}},Cint,Ptr{Ptr{cuDoubleComplex}},Cint,Ptr{cuDoubleComplex},Ptr{Ptr{cuDoubleComplex}},Cint,Cint),handle,transa,transb,m,n,k,alpha,Aarray,lda,Barray,ldb,beta,Carray,ldc,batchCount))
end

function cublasSgemmStridedBatched(handle,transa,transb,m,n,k,alpha,A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount)
    check_cublasstatus(ccall((:cublasSgemmStridedBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Clonglong,Ptr{Cfloat},Cint,Clonglong,Ptr{Cfloat},Ptr{Cfloat},Cint,Clonglong,Cint),handle,transa,transb,m,n,k,alpha,A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount))
end

function cublasDgemmStridedBatched(handle,transa,transb,m,n,k,alpha,A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount)
    check_cublasstatus(ccall((:cublasDgemmStridedBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Clonglong,Ptr{Cdouble},Cint,Clonglong,Ptr{Cdouble},Ptr{Cdouble},Cint,Clonglong,Cint),handle,transa,transb,m,n,k,alpha,A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount))
end

function cublasCgemmStridedBatched(handle,transa,transb,m,n,k,alpha,A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount)
    check_cublasstatus(ccall((:cublasCgemmStridedBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Clonglong,Ptr{cuComplex},Cint,Clonglong,Ptr{cuComplex},Ptr{cuComplex},Cint,Clonglong,Cint),handle,transa,transb,m,n,k,alpha,A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount))
end

function cublasCgemm3mStridedBatched(handle,transa,transb,m,n,k,alpha,A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount)
    check_cublasstatus(ccall((:cublasCgemm3mStridedBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Clonglong,Ptr{cuComplex},Cint,Clonglong,Ptr{cuComplex},Ptr{cuComplex},Cint,Clonglong,Cint),handle,transa,transb,m,n,k,alpha,A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount))
end

function cublasZgemmStridedBatched(handle,transa,transb,m,n,k,alpha,A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount)
    check_cublasstatus(ccall((:cublasZgemmStridedBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Clonglong,Ptr{cuDoubleComplex},Cint,Clonglong,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Clonglong,Cint),handle,transa,transb,m,n,k,alpha,A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount))
end

function cublasHgemmStridedBatched(handle,transa,transb,m,n,k,alpha,A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount)
    check_cublasstatus(ccall((:cublasHgemmStridedBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Cint,Ptr{__half},Ptr{__half},Cint,Clonglong,Ptr{__half},Cint,Clonglong,Ptr{__half},Ptr{__half},Cint,Clonglong,Cint),handle,transa,transb,m,n,k,alpha,A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount))
end

function cublasSgeam(handle,transa,transb,m,n,alpha,A,lda,beta,B,ldb,C,ldc)
    check_cublasstatus(ccall((:cublasSgeam,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,transa,transb,m,n,alpha,A,lda,beta,B,ldb,C,ldc))
end

function cublasDgeam(handle,transa,transb,m,n,alpha,A,lda,beta,B,ldb,C,ldc)
    check_cublasstatus(ccall((:cublasDgeam,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,transa,transb,m,n,alpha,A,lda,beta,B,ldb,C,ldc))
end

function cublasCgeam(handle,transa,transb,m,n,alpha,A,lda,beta,B,ldb,C,ldc)
    check_cublasstatus(ccall((:cublasCgeam,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,transa,transb,m,n,alpha,A,lda,beta,B,ldb,C,ldc))
end

function cublasZgeam(handle,transa,transb,m,n,alpha,A,lda,beta,B,ldb,C,ldc)
    check_cublasstatus(ccall((:cublasZgeam,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,cublasOperation_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,transa,transb,m,n,alpha,A,lda,beta,B,ldb,C,ldc))
end

function cublasSgetrfBatched(handle,n,A,lda,P,info,batchSize)
    check_cublasstatus(ccall((:cublasSgetrfBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Ptr{Cfloat}},Cint,Ptr{Cint},Ptr{Cint},Cint),handle,n,A,lda,P,info,batchSize))
end

function cublasDgetrfBatched(handle,n,A,lda,P,info,batchSize)
    check_cublasstatus(ccall((:cublasDgetrfBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Ptr{Cdouble}},Cint,Ptr{Cint},Ptr{Cint},Cint),handle,n,A,lda,P,info,batchSize))
end

function cublasCgetrfBatched(handle,n,A,lda,P,info,batchSize)
    check_cublasstatus(ccall((:cublasCgetrfBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Ptr{cuComplex}},Cint,Ptr{Cint},Ptr{Cint},Cint),handle,n,A,lda,P,info,batchSize))
end

function cublasZgetrfBatched(handle,n,A,lda,P,info,batchSize)
    check_cublasstatus(ccall((:cublasZgetrfBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Ptr{cuDoubleComplex}},Cint,Ptr{Cint},Ptr{Cint},Cint),handle,n,A,lda,P,info,batchSize))
end

function cublasSgetriBatched(handle,n,A,lda,P,C,ldc,info,batchSize)
    check_cublasstatus(ccall((:cublasSgetriBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Ptr{Cfloat}},Cint,Ptr{Cint},Ptr{Ptr{Cfloat}},Cint,Ptr{Cint},Cint),handle,n,A,lda,P,C,ldc,info,batchSize))
end

function cublasDgetriBatched(handle,n,A,lda,P,C,ldc,info,batchSize)
    check_cublasstatus(ccall((:cublasDgetriBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Ptr{Cdouble}},Cint,Ptr{Cint},Ptr{Ptr{Cdouble}},Cint,Ptr{Cint},Cint),handle,n,A,lda,P,C,ldc,info,batchSize))
end

function cublasCgetriBatched(handle,n,A,lda,P,C,ldc,info,batchSize)
    check_cublasstatus(ccall((:cublasCgetriBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Ptr{cuComplex}},Cint,Ptr{Cint},Ptr{Ptr{cuComplex}},Cint,Ptr{Cint},Cint),handle,n,A,lda,P,C,ldc,info,batchSize))
end

function cublasZgetriBatched(handle,n,A,lda,P,C,ldc,info,batchSize)
    check_cublasstatus(ccall((:cublasZgetriBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Ptr{cuDoubleComplex}},Cint,Ptr{Cint},Ptr{Ptr{cuDoubleComplex}},Cint,Ptr{Cint},Cint),handle,n,A,lda,P,C,ldc,info,batchSize))
end

function cublasSgetrsBatched(handle,trans,n,nrhs,Aarray,lda,devIpiv,Barray,ldb,info,batchSize)
    check_cublasstatus(ccall((:cublasSgetrsBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Ptr{Ptr{Cfloat}},Cint,Ptr{Cint},Ptr{Ptr{Cfloat}},Cint,Ptr{Cint},Cint),handle,trans,n,nrhs,Aarray,lda,devIpiv,Barray,ldb,info,batchSize))
end

function cublasDgetrsBatched(handle,trans,n,nrhs,Aarray,lda,devIpiv,Barray,ldb,info,batchSize)
    check_cublasstatus(ccall((:cublasDgetrsBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Ptr{Ptr{Cdouble}},Cint,Ptr{Cint},Ptr{Ptr{Cdouble}},Cint,Ptr{Cint},Cint),handle,trans,n,nrhs,Aarray,lda,devIpiv,Barray,ldb,info,batchSize))
end

function cublasCgetrsBatched(handle,trans,n,nrhs,Aarray,lda,devIpiv,Barray,ldb,info,batchSize)
    check_cublasstatus(ccall((:cublasCgetrsBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Ptr{Ptr{cuComplex}},Cint,Ptr{Cint},Ptr{Ptr{cuComplex}},Cint,Ptr{Cint},Cint),handle,trans,n,nrhs,Aarray,lda,devIpiv,Barray,ldb,info,batchSize))
end

function cublasZgetrsBatched(handle,trans,n,nrhs,Aarray,lda,devIpiv,Barray,ldb,info,batchSize)
    check_cublasstatus(ccall((:cublasZgetrsBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Ptr{Ptr{cuDoubleComplex}},Cint,Ptr{Cint},Ptr{Ptr{cuDoubleComplex}},Cint,Ptr{Cint},Cint),handle,trans,n,nrhs,Aarray,lda,devIpiv,Barray,ldb,info,batchSize))
end

function cublasStrsmBatched(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,batchCount)
    check_cublasstatus(ccall((:cublasStrsmBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{Cfloat},Ptr{Ptr{Cfloat}},Cint,Ptr{Ptr{Cfloat}},Cint,Cint),handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,batchCount))
end

function cublasDtrsmBatched(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,batchCount)
    check_cublasstatus(ccall((:cublasDtrsmBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{Cdouble},Ptr{Ptr{Cdouble}},Cint,Ptr{Ptr{Cdouble}},Cint,Cint),handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,batchCount))
end

function cublasCtrsmBatched(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,batchCount)
    check_cublasstatus(ccall((:cublasCtrsmBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{cuComplex},Ptr{Ptr{cuComplex}},Cint,Ptr{Ptr{cuComplex}},Cint,Cint),handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,batchCount))
end

function cublasZtrsmBatched(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,batchCount)
    check_cublasstatus(ccall((:cublasZtrsmBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,Cint,Cint,Ptr{cuDoubleComplex},Ptr{Ptr{cuDoubleComplex}},Cint,Ptr{Ptr{cuDoubleComplex}},Cint,Cint),handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,batchCount))
end

function cublasSmatinvBatched(handle,n,A,lda,Ainv,lda_inv,info,batchSize)
    check_cublasstatus(ccall((:cublasSmatinvBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Ptr{Cfloat}},Cint,Ptr{Ptr{Cfloat}},Cint,Ptr{Cint},Cint),handle,n,A,lda,Ainv,lda_inv,info,batchSize))
end

function cublasDmatinvBatched(handle,n,A,lda,Ainv,lda_inv,info,batchSize)
    check_cublasstatus(ccall((:cublasDmatinvBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Ptr{Cdouble}},Cint,Ptr{Ptr{Cdouble}},Cint,Ptr{Cint},Cint),handle,n,A,lda,Ainv,lda_inv,info,batchSize))
end

function cublasCmatinvBatched(handle,n,A,lda,Ainv,lda_inv,info,batchSize)
    check_cublasstatus(ccall((:cublasCmatinvBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Ptr{cuComplex}},Cint,Ptr{Ptr{cuComplex}},Cint,Ptr{Cint},Cint),handle,n,A,lda,Ainv,lda_inv,info,batchSize))
end

function cublasZmatinvBatched(handle,n,A,lda,Ainv,lda_inv,info,batchSize)
    check_cublasstatus(ccall((:cublasZmatinvBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Ptr{Ptr{cuDoubleComplex}},Cint,Ptr{Ptr{cuDoubleComplex}},Cint,Ptr{Cint},Cint),handle,n,A,lda,Ainv,lda_inv,info,batchSize))
end

function cublasSgeqrfBatched(handle,m,n,Aarray,lda,TauArray,info,batchSize)
    check_cublasstatus(ccall((:cublasSgeqrfBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Cint,Ptr{Ptr{Cfloat}},Cint,Ptr{Ptr{Cfloat}},Ptr{Cint},Cint),handle,m,n,Aarray,lda,TauArray,info,batchSize))
end

function cublasDgeqrfBatched(handle,m,n,Aarray,lda,TauArray,info,batchSize)
    check_cublasstatus(ccall((:cublasDgeqrfBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Cint,Ptr{Ptr{Cdouble}},Cint,Ptr{Ptr{Cdouble}},Ptr{Cint},Cint),handle,m,n,Aarray,lda,TauArray,info,batchSize))
end

function cublasCgeqrfBatched(handle,m,n,Aarray,lda,TauArray,info,batchSize)
    check_cublasstatus(ccall((:cublasCgeqrfBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Cint,Ptr{Ptr{cuComplex}},Cint,Ptr{Ptr{cuComplex}},Ptr{Cint},Cint),handle,m,n,Aarray,lda,TauArray,info,batchSize))
end

function cublasZgeqrfBatched(handle,m,n,Aarray,lda,TauArray,info,batchSize)
    check_cublasstatus(ccall((:cublasZgeqrfBatched,libcublas),cublasStatus_t,(cublasHandle_t,Cint,Cint,Ptr{Ptr{cuDoubleComplex}},Cint,Ptr{Ptr{cuDoubleComplex}},Ptr{Cint},Cint),handle,m,n,Aarray,lda,TauArray,info,batchSize))
end

function cublasSgelsBatched(handle,trans,m,n,nrhs,Aarray,lda,Carray,ldc,info,devInfoArray,batchSize)
    check_cublasstatus(ccall((:cublasSgelsBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Cint,Ptr{Ptr{Cfloat}},Cint,Ptr{Ptr{Cfloat}},Cint,Ptr{Cint},Ptr{Cint},Cint),handle,trans,m,n,nrhs,Aarray,lda,Carray,ldc,info,devInfoArray,batchSize))
end

function cublasDgelsBatched(handle,trans,m,n,nrhs,Aarray,lda,Carray,ldc,info,devInfoArray,batchSize)
    check_cublasstatus(ccall((:cublasDgelsBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Cint,Ptr{Ptr{Cdouble}},Cint,Ptr{Ptr{Cdouble}},Cint,Ptr{Cint},Ptr{Cint},Cint),handle,trans,m,n,nrhs,Aarray,lda,Carray,ldc,info,devInfoArray,batchSize))
end

function cublasCgelsBatched(handle,trans,m,n,nrhs,Aarray,lda,Carray,ldc,info,devInfoArray,batchSize)
    check_cublasstatus(ccall((:cublasCgelsBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Cint,Ptr{Ptr{cuComplex}},Cint,Ptr{Ptr{cuComplex}},Cint,Ptr{Cint},Ptr{Cint},Cint),handle,trans,m,n,nrhs,Aarray,lda,Carray,ldc,info,devInfoArray,batchSize))
end

function cublasZgelsBatched(handle,trans,m,n,nrhs,Aarray,lda,Carray,ldc,info,devInfoArray,batchSize)
    check_cublasstatus(ccall((:cublasZgelsBatched,libcublas),cublasStatus_t,(cublasHandle_t,cublasOperation_t,Cint,Cint,Cint,Ptr{Ptr{cuDoubleComplex}},Cint,Ptr{Ptr{cuDoubleComplex}},Cint,Ptr{Cint},Ptr{Cint},Cint),handle,trans,m,n,nrhs,Aarray,lda,Carray,ldc,info,devInfoArray,batchSize))
end

function cublasSdgmm(handle,mode,m,n,A,lda,x,incx,C,ldc)
    check_cublasstatus(ccall((:cublasSdgmm,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,Cint,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Cint),handle,mode,m,n,A,lda,x,incx,C,ldc))
end

function cublasDdgmm(handle,mode,m,n,A,lda,x,incx,C,ldc)
    check_cublasstatus(ccall((:cublasDdgmm,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,Cint,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint),handle,mode,m,n,A,lda,x,incx,C,ldc))
end

function cublasCdgmm(handle,mode,m,n,A,lda,x,incx,C,ldc)
    check_cublasstatus(ccall((:cublasCdgmm,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,Cint,Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint,Ptr{cuComplex},Cint),handle,mode,m,n,A,lda,x,incx,C,ldc))
end

function cublasZdgmm(handle,mode,m,n,A,lda,x,incx,C,ldc)
    check_cublasstatus(ccall((:cublasZdgmm,libcublas),cublasStatus_t,(cublasHandle_t,cublasSideMode_t,Cint,Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex},Cint),handle,mode,m,n,A,lda,x,incx,C,ldc))
end

function cublasStpttr(handle,uplo,n,AP,A,lda)
    check_cublasstatus(ccall((:cublasStpttr,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cfloat},Ptr{Cfloat},Cint),handle,uplo,n,AP,A,lda))
end

function cublasDtpttr(handle,uplo,n,AP,A,lda)
    check_cublasstatus(ccall((:cublasDtpttr,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cdouble},Ptr{Cdouble},Cint),handle,uplo,n,AP,A,lda))
end

function cublasCtpttr(handle,uplo,n,AP,A,lda)
    check_cublasstatus(ccall((:cublasCtpttr,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuComplex},Ptr{cuComplex},Cint),handle,uplo,n,AP,A,lda))
end

function cublasZtpttr(handle,uplo,n,AP,A,lda)
    check_cublasstatus(ccall((:cublasZtpttr,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuDoubleComplex},Ptr{cuDoubleComplex},Cint),handle,uplo,n,AP,A,lda))
end

function cublasStrttp(handle,uplo,n,A,lda,AP)
    check_cublasstatus(ccall((:cublasStrttp,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat}),handle,uplo,n,A,lda,AP))
end

function cublasDtrttp(handle,uplo,n,A,lda,AP)
    check_cublasstatus(ccall((:cublasDtrttp,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble}),handle,uplo,n,A,lda,AP))
end

function cublasCtrttp(handle,uplo,n,A,lda,AP)
    check_cublasstatus(ccall((:cublasCtrttp,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuComplex},Cint,Ptr{cuComplex}),handle,uplo,n,A,lda,AP))
end

function cublasZtrttp(handle,uplo,n,A,lda,AP)
    check_cublasstatus(ccall((:cublasZtrttp,libcublas),cublasStatus_t,(cublasHandle_t,cublasFillMode_t,Cint,Ptr{cuDoubleComplex},Cint,Ptr{cuDoubleComplex}),handle,uplo,n,A,lda,AP))
end
