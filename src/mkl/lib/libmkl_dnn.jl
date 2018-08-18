# Julia wrapper for header: /opt/intel/compilers_and_libraries_2017.1.132/linux/mkl/include/mkl_dnn.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function dnnLayoutCreate_F32(pLayout,dimension,size,strides)
    check_dnnerror(ccall((:dnnLayoutCreate_F32,libmkl),dnnError_t,(Ptr{dnnLayout_t},Csize_t,Ptr{Csize_t},Ptr{Csize_t}),pLayout,dimension,size,strides))
end

function dnnLayoutCreateFromPrimitive_F32(pLayout,primitive,_type)
    check_dnnerror(ccall((:dnnLayoutCreateFromPrimitive_F32,libmkl),dnnError_t,(Ptr{dnnLayout_t},dnnPrimitive_t,dnnResourceType_t),pLayout,primitive,_type))
end

function dnnLayoutGetMemorySize_F32(layout)
    ccall((:dnnLayoutGetMemorySize_F32,libmkl),Csize_t,(dnnLayout_t,),layout)
end

function dnnLayoutCompare_F32(l1,l2)
    ccall((:dnnLayoutCompare_F32,libmkl),Cint,(dnnLayout_t,dnnLayout_t),l1,l2)
end

function dnnAllocateBuffer_F32(pPtr,layout)
    check_dnnerror(ccall((:dnnAllocateBuffer_F32,libmkl),dnnError_t,(Ptr{Ptr{Cvoid}},dnnLayout_t),pPtr,layout))
end

function dnnReleaseBuffer_F32(ptr)
    check_dnnerror(ccall((:dnnReleaseBuffer_F32,libmkl),dnnError_t,(Ptr{Cvoid},),ptr))
end

function dnnLayoutDelete_F32(layout)
    check_dnnerror(ccall((:dnnLayoutDelete_F32,libmkl),dnnError_t,(dnnLayout_t,),layout))
end

function dnnPrimitiveAttributesCreate_F32(attributes)
    check_dnnerror(ccall((:dnnPrimitiveAttributesCreate_F32,libmkl),dnnError_t,(Ptr{dnnPrimitiveAttributes_t},),attributes))
end

function dnnPrimitiveAttributesDestroy_F32(attributes)
    check_dnnerror(ccall((:dnnPrimitiveAttributesDestroy_F32,libmkl),dnnError_t,(dnnPrimitiveAttributes_t,),attributes))
end

function dnnPrimitiveGetAttributes_F32(primitive,attributes)
    check_dnnerror(ccall((:dnnPrimitiveGetAttributes_F32,libmkl),dnnError_t,(dnnPrimitive_t,Ptr{dnnPrimitiveAttributes_t}),primitive,attributes))
end

function dnnExecute_F32(primitive,resources)
    check_dnnerror(ccall((:dnnExecute_F32,libmkl),dnnError_t,(dnnPrimitive_t,Ptr{Ptr{Cvoid}}),primitive,resources))
end

function dnnExecuteAsync_F32(primitive,resources)
    check_dnnerror(ccall((:dnnExecuteAsync_F32,libmkl),dnnError_t,(dnnPrimitive_t,Ptr{Ptr{Cvoid}}),primitive,resources))
end

function dnnWaitFor_F32(primitive)
    check_dnnerror(ccall((:dnnWaitFor_F32,libmkl),dnnError_t,(dnnPrimitive_t,),primitive))
end

function dnnDelete_F32(primitive)
    check_dnnerror(ccall((:dnnDelete_F32,libmkl),dnnError_t,(dnnPrimitive_t,),primitive))
end

function dnnConversionCreate_F32(pConversion,from,to)
    check_dnnerror(ccall((:dnnConversionCreate_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnLayout_t,dnnLayout_t),pConversion,from,to))
end

function dnnConversionExecute_F32(conversion,from,to)
    check_dnnerror(ccall((:dnnConversionExecute_F32,libmkl),dnnError_t,(dnnPrimitive_t,Ptr{Cvoid},Ptr{Cvoid}),conversion,from,to))
end

function dnnSumCreate_F32(pSum,attributes,nSummands,layout,coefficients)
    check_dnnerror(ccall((:dnnSumCreate_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,dnnLayout_t,Ptr{Cfloat}),pSum,attributes,nSummands,layout,coefficients))
end

function dnnConcatCreate_F32(pConcat,attributes,nSrcTensors,src)
    check_dnnerror(ccall((:dnnConcatCreate_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,Ptr{dnnLayout_t}),pConcat,attributes,nSrcTensors,src))
end

function dnnSplitCreate_F32(pSplit,attributes,nDstTensors,layout,dstChannelSize)
    check_dnnerror(ccall((:dnnSplitCreate_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,dnnLayout_t,Ptr{Csize_t}),pSplit,attributes,nDstTensors,layout,dstChannelSize))
end

function dnnScaleCreate_F32(pScale,attributes,dataLayout,alpha)
    check_dnnerror(ccall((:dnnScaleCreate_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,Cfloat),pScale,attributes,dataLayout,alpha))
end

function dnnConvolutionCreateForward_F32(pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnConvolutionCreateForward_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnConvolutionCreateForwardBias_F32(pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnConvolutionCreateForwardBias_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnConvolutionCreateBackwardData_F32(pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnConvolutionCreateBackwardData_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnConvolutionCreateBackwardFilter_F32(pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnConvolutionCreateBackwardFilter_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnConvolutionCreateBackwardBias_F32(pConvolution,attributes,algorithm,dimension,dstSize)
    check_dnnerror(ccall((:dnnConvolutionCreateBackwardBias_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Ptr{Csize_t}),pConvolution,attributes,algorithm,dimension,dstSize))
end

function dnnGroupsConvolutionCreateForward_F32(pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnGroupsConvolutionCreateForward_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnGroupsConvolutionCreateForwardBias_F32(pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnGroupsConvolutionCreateForwardBias_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnGroupsConvolutionCreateBackwardData_F32(pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnGroupsConvolutionCreateBackwardData_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnGroupsConvolutionCreateBackwardFilter_F32(pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnGroupsConvolutionCreateBackwardFilter_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnGroupsConvolutionCreateBackwardBias_F32(pConvolution,attributes,algorithm,groups,dimension,dstSize)
    check_dnnerror(ccall((:dnnGroupsConvolutionCreateBackwardBias_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Csize_t,Ptr{Csize_t}),pConvolution,attributes,algorithm,groups,dimension,dstSize))
end

function dnnReLUCreateForward_F32(pRelu,attributes,dataLayout,negativeSlope)
    check_dnnerror(ccall((:dnnReLUCreateForward_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,Cfloat),pRelu,attributes,dataLayout,negativeSlope))
end

function dnnReLUCreateBackward_F32(pRelu,attributes,diffLayout,dataLayout,negativeSlope)
    check_dnnerror(ccall((:dnnReLUCreateBackward_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,dnnLayout_t,Cfloat),pRelu,attributes,diffLayout,dataLayout,negativeSlope))
end

function dnnLRNCreateForward_F32(pLrn,attributes,dataLayout,kernel_size,alpha,beta,k)
    check_dnnerror(ccall((:dnnLRNCreateForward_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,Csize_t,Cfloat,Cfloat,Cfloat),pLrn,attributes,dataLayout,kernel_size,alpha,beta,k))
end

function dnnLRNCreateBackward_F32(pLrn,attributes,diffLayout,dataLayout,kernel_size,alpha,beta,k)
    check_dnnerror(ccall((:dnnLRNCreateBackward_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,dnnLayout_t,Csize_t,Cfloat,Cfloat,Cfloat),pLrn,attributes,diffLayout,dataLayout,kernel_size,alpha,beta,k))
end

function dnnBatchNormalizationCreateForward_F32(pBatchNormalization,attributes,dataLayout,eps)
    check_dnnerror(ccall((:dnnBatchNormalizationCreateForward_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,Cfloat),pBatchNormalization,attributes,dataLayout,eps))
end

function dnnBatchNormalizationCreateBackwardScaleShift_F32(pBatchNormalization,attributes,dataLayout,eps)
    check_dnnerror(ccall((:dnnBatchNormalizationCreateBackwardScaleShift_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,Cfloat),pBatchNormalization,attributes,dataLayout,eps))
end

function dnnBatchNormalizationCreateBackwardData_F32(pBatchNormalization,attributes,dataLayout,eps)
    check_dnnerror(ccall((:dnnBatchNormalizationCreateBackwardData_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,Cfloat),pBatchNormalization,attributes,dataLayout,eps))
end

function dnnPoolingCreateForward_F32(pPooling,attributes,op,srcLayout,kernelSize,kernelStride,inputOffset,borderType)
    check_dnnerror(ccall((:dnnPoolingCreateForward_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,dnnLayout_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pPooling,attributes,op,srcLayout,kernelSize,kernelStride,inputOffset,borderType))
end

function dnnPoolingCreateBackward_F32(pPooling,attributes,op,srcLayout,kernelSize,kernelStride,inputOffset,borderType)
    check_dnnerror(ccall((:dnnPoolingCreateBackward_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,dnnLayout_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pPooling,attributes,op,srcLayout,kernelSize,kernelStride,inputOffset,borderType))
end

function dnnInnerProductCreateForward_F32(pInnerProduct,attributes,dimensions,srcSize,outputChannels)
    check_dnnerror(ccall((:dnnInnerProductCreateForward_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,Ptr{Csize_t},Csize_t),pInnerProduct,attributes,dimensions,srcSize,outputChannels))
end

function dnnInnerProductCreateForwardBias_F32(pInnerProduct,attributes,dimensions,srcSize,outputChannels)
    check_dnnerror(ccall((:dnnInnerProductCreateForwardBias_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,Ptr{Csize_t},Csize_t),pInnerProduct,attributes,dimensions,srcSize,outputChannels))
end

function dnnInnerProductCreateBackwardData_F32(pInnerProduct,attributes,dimensions,srcSize,outputChannels)
    check_dnnerror(ccall((:dnnInnerProductCreateBackwardData_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,Ptr{Csize_t},Csize_t),pInnerProduct,attributes,dimensions,srcSize,outputChannels))
end

function dnnInnerProductCreateBackwardFilter_F32(pInnerProduct,attributes,dimensions,srcSize,outputChannels)
    check_dnnerror(ccall((:dnnInnerProductCreateBackwardFilter_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,Ptr{Csize_t},Csize_t),pInnerProduct,attributes,dimensions,srcSize,outputChannels))
end

function dnnInnerProductCreateBackwardBias_F32(pInnerProduct,attributes,dimensions,dstSize)
    check_dnnerror(ccall((:dnnInnerProductCreateBackwardBias_F32,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,Ptr{Csize_t}),pInnerProduct,attributes,dimensions,dstSize))
end

function dnnLayoutCreate_F64(pLayout,dimension,size,strides)
    check_dnnerror(ccall((:dnnLayoutCreate_F64,libmkl),dnnError_t,(Ptr{dnnLayout_t},Csize_t,Ptr{Csize_t},Ptr{Csize_t}),pLayout,dimension,size,strides))
end

function dnnLayoutCreateFromPrimitive_F64(pLayout,primitive,_type)
    check_dnnerror(ccall((:dnnLayoutCreateFromPrimitive_F64,libmkl),dnnError_t,(Ptr{dnnLayout_t},dnnPrimitive_t,dnnResourceType_t),pLayout,primitive,_type))
end

function dnnLayoutGetMemorySize_F64(layout)
    ccall((:dnnLayoutGetMemorySize_F64,libmkl),Csize_t,(dnnLayout_t,),layout)
end

function dnnLayoutCompare_F64(l1,l2)
    ccall((:dnnLayoutCompare_F64,libmkl),Cint,(dnnLayout_t,dnnLayout_t),l1,l2)
end

function dnnAllocateBuffer_F64(pPtr,layout)
    check_dnnerror(ccall((:dnnAllocateBuffer_F64,libmkl),dnnError_t,(Ptr{Ptr{Cvoid}},dnnLayout_t),pPtr,layout))
end

function dnnReleaseBuffer_F64(ptr)
    check_dnnerror(ccall((:dnnReleaseBuffer_F64,libmkl),dnnError_t,(Ptr{Cvoid},),ptr))
end

function dnnLayoutDelete_F64(layout)
    check_dnnerror(ccall((:dnnLayoutDelete_F64,libmkl),dnnError_t,(dnnLayout_t,),layout))
end

function dnnPrimitiveAttributesCreate_F64(attributes)
    check_dnnerror(ccall((:dnnPrimitiveAttributesCreate_F64,libmkl),dnnError_t,(Ptr{dnnPrimitiveAttributes_t},),attributes))
end

function dnnPrimitiveAttributesDestroy_F64(attributes)
    check_dnnerror(ccall((:dnnPrimitiveAttributesDestroy_F64,libmkl),dnnError_t,(dnnPrimitiveAttributes_t,),attributes))
end

function dnnPrimitiveGetAttributes_F64(primitive,attributes)
    check_dnnerror(ccall((:dnnPrimitiveGetAttributes_F64,libmkl),dnnError_t,(dnnPrimitive_t,Ptr{dnnPrimitiveAttributes_t}),primitive,attributes))
end

function dnnExecute_F64(primitive,resources)
    check_dnnerror(ccall((:dnnExecute_F64,libmkl),dnnError_t,(dnnPrimitive_t,Ptr{Ptr{Cvoid}}),primitive,resources))
end

function dnnExecuteAsync_F64(primitive,resources)
    check_dnnerror(ccall((:dnnExecuteAsync_F64,libmkl),dnnError_t,(dnnPrimitive_t,Ptr{Ptr{Cvoid}}),primitive,resources))
end

function dnnWaitFor_F64(primitive)
    check_dnnerror(ccall((:dnnWaitFor_F64,libmkl),dnnError_t,(dnnPrimitive_t,),primitive))
end

function dnnDelete_F64(primitive)
    check_dnnerror(ccall((:dnnDelete_F64,libmkl),dnnError_t,(dnnPrimitive_t,),primitive))
end

function dnnConversionCreate_F64(pConversion,from,to)
    check_dnnerror(ccall((:dnnConversionCreate_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnLayout_t,dnnLayout_t),pConversion,from,to))
end

function dnnConversionExecute_F64(conversion,from,to)
    check_dnnerror(ccall((:dnnConversionExecute_F64,libmkl),dnnError_t,(dnnPrimitive_t,Ptr{Cvoid},Ptr{Cvoid}),conversion,from,to))
end

function dnnSumCreate_F64(pSum,attributes,nSummands,layout,coefficients)
    check_dnnerror(ccall((:dnnSumCreate_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,dnnLayout_t,Ptr{Cdouble}),pSum,attributes,nSummands,layout,coefficients))
end

function dnnConcatCreate_F64(pConcat,attributes,nSrcTensors,src)
    check_dnnerror(ccall((:dnnConcatCreate_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,Ptr{dnnLayout_t}),pConcat,attributes,nSrcTensors,src))
end

function dnnSplitCreate_F64(pSplit,attributes,nDstTensors,layout,dstChannelSize)
    check_dnnerror(ccall((:dnnSplitCreate_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,dnnLayout_t,Ptr{Csize_t}),pSplit,attributes,nDstTensors,layout,dstChannelSize))
end

function dnnScaleCreate_F64(pScale,attributes,dataLayout,alpha)
    check_dnnerror(ccall((:dnnScaleCreate_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,Cdouble),pScale,attributes,dataLayout,alpha))
end

function dnnConvolutionCreateForward_F64(pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnConvolutionCreateForward_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnConvolutionCreateForwardBias_F64(pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnConvolutionCreateForwardBias_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnConvolutionCreateBackwardData_F64(pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnConvolutionCreateBackwardData_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnConvolutionCreateBackwardFilter_F64(pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnConvolutionCreateBackwardFilter_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnConvolutionCreateBackwardBias_F64(pConvolution,attributes,algorithm,dimension,dstSize)
    check_dnnerror(ccall((:dnnConvolutionCreateBackwardBias_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Ptr{Csize_t}),pConvolution,attributes,algorithm,dimension,dstSize))
end

function dnnGroupsConvolutionCreateForward_F64(pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnGroupsConvolutionCreateForward_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnGroupsConvolutionCreateForwardBias_F64(pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnGroupsConvolutionCreateForwardBias_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnGroupsConvolutionCreateBackwardData_F64(pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnGroupsConvolutionCreateBackwardData_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnGroupsConvolutionCreateBackwardFilter_F64(pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType)
    check_dnnerror(ccall((:dnnGroupsConvolutionCreateBackwardFilter_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Csize_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pConvolution,attributes,algorithm,groups,dimension,srcSize,dstSize,filterSize,convolutionStrides,inputOffset,borderType))
end

function dnnGroupsConvolutionCreateBackwardBias_F64(pConvolution,attributes,algorithm,groups,dimension,dstSize)
    check_dnnerror(ccall((:dnnGroupsConvolutionCreateBackwardBias_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,Csize_t,Csize_t,Ptr{Csize_t}),pConvolution,attributes,algorithm,groups,dimension,dstSize))
end

function dnnReLUCreateForward_F64(pRelu,attributes,dataLayout,negativeSlope)
    check_dnnerror(ccall((:dnnReLUCreateForward_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,Cdouble),pRelu,attributes,dataLayout,negativeSlope))
end

function dnnReLUCreateBackward_F64(pRelu,attributes,diffLayout,dataLayout,negativeSlope)
    check_dnnerror(ccall((:dnnReLUCreateBackward_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,dnnLayout_t,Cdouble),pRelu,attributes,diffLayout,dataLayout,negativeSlope))
end

function dnnLRNCreateForward_F64(pLrn,attributes,dataLayout,kernel_size,alpha,beta,k)
    check_dnnerror(ccall((:dnnLRNCreateForward_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,Csize_t,Cdouble,Cdouble,Cdouble),pLrn,attributes,dataLayout,kernel_size,alpha,beta,k))
end

function dnnLRNCreateBackward_F64(pLrn,attributes,diffLayout,dataLayout,kernel_size,alpha,beta,k)
    check_dnnerror(ccall((:dnnLRNCreateBackward_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,dnnLayout_t,Csize_t,Cdouble,Cdouble,Cdouble),pLrn,attributes,diffLayout,dataLayout,kernel_size,alpha,beta,k))
end

function dnnBatchNormalizationCreateForward_F64(pBatchNormalization,attributes,dataLayout,eps)
    check_dnnerror(ccall((:dnnBatchNormalizationCreateForward_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,Cdouble),pBatchNormalization,attributes,dataLayout,eps))
end

function dnnBatchNormalizationCreateBackwardScaleShift_F64(pBatchNormalization,attributes,dataLayout,eps)
    check_dnnerror(ccall((:dnnBatchNormalizationCreateBackwardScaleShift_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,Cdouble),pBatchNormalization,attributes,dataLayout,eps))
end

function dnnBatchNormalizationCreateBackwardData_F64(pBatchNormalization,attributes,dataLayout,eps)
    check_dnnerror(ccall((:dnnBatchNormalizationCreateBackwardData_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnLayout_t,Cdouble),pBatchNormalization,attributes,dataLayout,eps))
end

function dnnPoolingCreateForward_F64(pPooling,attributes,op,srcLayout,kernelSize,kernelStride,inputOffset,borderType)
    check_dnnerror(ccall((:dnnPoolingCreateForward_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,dnnLayout_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pPooling,attributes,op,srcLayout,kernelSize,kernelStride,inputOffset,borderType))
end

function dnnPoolingCreateBackward_F64(pPooling,attributes,op,srcLayout,kernelSize,kernelStride,inputOffset,borderType)
    check_dnnerror(ccall((:dnnPoolingCreateBackward_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,dnnAlgorithm_t,dnnLayout_t,Ptr{Csize_t},Ptr{Csize_t},Ptr{Cint},dnnBorder_t),pPooling,attributes,op,srcLayout,kernelSize,kernelStride,inputOffset,borderType))
end

function dnnInnerProductCreateForward_F64(pInnerProduct,attributes,dimensions,srcSize,outputChannels)
    check_dnnerror(ccall((:dnnInnerProductCreateForward_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,Ptr{Csize_t},Csize_t),pInnerProduct,attributes,dimensions,srcSize,outputChannels))
end

function dnnInnerProductCreateForwardBias_F64(pInnerProduct,attributes,dimensions,srcSize,outputChannels)
    check_dnnerror(ccall((:dnnInnerProductCreateForwardBias_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,Ptr{Csize_t},Csize_t),pInnerProduct,attributes,dimensions,srcSize,outputChannels))
end

function dnnInnerProductCreateBackwardData_F64(pInnerProduct,attributes,dimensions,srcSize,outputChannels)
    check_dnnerror(ccall((:dnnInnerProductCreateBackwardData_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,Ptr{Csize_t},Csize_t),pInnerProduct,attributes,dimensions,srcSize,outputChannels))
end

function dnnInnerProductCreateBackwardFilter_F64(pInnerProduct,attributes,dimensions,srcSize,outputChannels)
    check_dnnerror(ccall((:dnnInnerProductCreateBackwardFilter_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,Ptr{Csize_t},Csize_t),pInnerProduct,attributes,dimensions,srcSize,outputChannels))
end

function dnnInnerProductCreateBackwardBias_F64(pInnerProduct,attributes,dimensions,dstSize)
    check_dnnerror(ccall((:dnnInnerProductCreateBackwardBias_F64,libmkl),dnnError_t,(Ptr{dnnPrimitive_t},dnnPrimitiveAttributes_t,Csize_t,Ptr{Csize_t}),pInnerProduct,attributes,dimensions,dstSize))
end
