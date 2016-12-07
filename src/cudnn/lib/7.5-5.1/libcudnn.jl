# Julia wrapper for header: /home/shindo/local-lemon/cuda-7.5/include/cudnn.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function cudnnGetVersion()
    ccall((:cudnnGetVersion,libcudnn),Csize_t,())
end

function cudnnGetErrorString(status)
    ccall((:cudnnGetErrorString,libcudnn),Ptr{UInt8},(cudnnStatus_t,),status)
end

function cudnnCreate(handle)
    checkstatus(ccall((:cudnnCreate,libcudnn),cudnnStatus_t,(Ptr{cudnnHandle_t},),handle))
end

function cudnnDestroy(handle)
    checkstatus(ccall((:cudnnDestroy,libcudnn),cudnnStatus_t,(cudnnHandle_t,),handle))
end

function cudnnSetStream(handle,streamId)
    checkstatus(ccall((:cudnnSetStream,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudaStream_t),handle,streamId))
end

function cudnnGetStream(handle,streamId)
    checkstatus(ccall((:cudnnGetStream,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{cudaStream_t}),handle,streamId))
end

function cudnnCreateTensorDescriptor(tensorDesc)
    checkstatus(ccall((:cudnnCreateTensorDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnTensorDescriptor_t},),tensorDesc))
end

function cudnnSetTensor4dDescriptor(tensorDesc,format,dataType,n,c,h,w)
    checkstatus(ccall((:cudnnSetTensor4dDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnTensorFormat_t,cudnnDataType_t,Cint,Cint,Cint,Cint),tensorDesc,format,dataType,n,c,h,w))
end

function cudnnSetTensor4dDescriptorEx(tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride)
    checkstatus(ccall((:cudnnSetTensor4dDescriptorEx,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnDataType_t,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride))
end

function cudnnGetTensor4dDescriptor(tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride)
    checkstatus(ccall((:cudnnGetTensor4dDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride))
end

function cudnnSetTensorNdDescriptor(tensorDesc,dataType,nbDims,dimA,strideA)
    checkstatus(ccall((:cudnnSetTensorNdDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnDataType_t,Cint,Ptr{Cint},Ptr{Cint}),tensorDesc,dataType,nbDims,dimA,strideA))
end

function cudnnGetTensorNdDescriptor(tensorDesc,nbDimsRequested,dataType,nbDims,dimA,strideA)
    checkstatus(ccall((:cudnnGetTensorNdDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,nbDimsRequested,dataType,nbDims,dimA,strideA))
end

function cudnnDestroyTensorDescriptor(tensorDesc)
    checkstatus(ccall((:cudnnDestroyTensorDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,),tensorDesc))
end

function cudnnTransformTensor(handle,alpha,xDesc,x,beta,yDesc,y)
    checkstatus(ccall((:cudnnTransformTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,xDesc,x,beta,yDesc,y))
end

function cudnnAddTensor(handle,alpha,aDesc,A,beta,cDesc,C)
    checkstatus(ccall((:cudnnAddTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,aDesc,A,beta,cDesc,C))
end

function cudnnCreateOpTensorDescriptor(opTensorDesc)
    checkstatus(ccall((:cudnnCreateOpTensorDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnOpTensorDescriptor_t},),opTensorDesc))
end

function cudnnSetOpTensorDescriptor(opTensorDesc,opTensorOp,opTensorCompType,opTensorNanOpt)
    checkstatus(ccall((:cudnnSetOpTensorDescriptor,libcudnn),cudnnStatus_t,(cudnnOpTensorDescriptor_t,cudnnOpTensorOp_t,cudnnDataType_t,cudnnNanPropagation_t),opTensorDesc,opTensorOp,opTensorCompType,opTensorNanOpt))
end

function cudnnGetOpTensorDescriptor(opTensorDesc,opTensorOp,opTensorCompType,opTensorNanOpt)
    checkstatus(ccall((:cudnnGetOpTensorDescriptor,libcudnn),cudnnStatus_t,(cudnnOpTensorDescriptor_t,Ptr{cudnnOpTensorOp_t},Ptr{cudnnDataType_t},Ptr{cudnnNanPropagation_t}),opTensorDesc,opTensorOp,opTensorCompType,opTensorNanOpt))
end

function cudnnDestroyOpTensorDescriptor(opTensorDesc)
    checkstatus(ccall((:cudnnDestroyOpTensorDescriptor,libcudnn),cudnnStatus_t,(cudnnOpTensorDescriptor_t,),opTensorDesc))
end

function cudnnOpTensor(handle,opTensorDesc,alpha1,aDesc,A,alpha2,bDesc,B,beta,cDesc,C)
    checkstatus(ccall((:cudnnOpTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnOpTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,opTensorDesc,alpha1,aDesc,A,alpha2,bDesc,B,beta,cDesc,C))
end

function cudnnSetTensor(handle,yDesc,y,valuePtr)
    checkstatus(ccall((:cudnnSetTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,yDesc,y,valuePtr))
end

function cudnnScaleTensor(handle,yDesc,y,alpha)
    checkstatus(ccall((:cudnnScaleTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,yDesc,y,alpha))
end

function cudnnCreateFilterDescriptor(filterDesc)
    checkstatus(ccall((:cudnnCreateFilterDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnFilterDescriptor_t},),filterDesc))
end

function cudnnSetFilter4dDescriptor(filterDesc,dataType,format,k,c,h,w)
    checkstatus(ccall((:cudnnSetFilter4dDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,cudnnTensorFormat_t,Cint,Cint,Cint,Cint),filterDesc,dataType,format,k,c,h,w))
end

function cudnnGetFilter4dDescriptor(filterDesc,dataType,format,k,c,h,w)
    checkstatus(ccall((:cudnnGetFilter4dDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Ptr{cudnnDataType_t},Ptr{cudnnTensorFormat_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),filterDesc,dataType,format,k,c,h,w))
end

function cudnnSetFilterNdDescriptor(filterDesc,dataType,format,nbDims,filterDimA)
    checkstatus(ccall((:cudnnSetFilterNdDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,cudnnTensorFormat_t,Cint,Ptr{Cint}),filterDesc,dataType,format,nbDims,filterDimA))
end

function cudnnGetFilterNdDescriptor(filterDesc,nbDimsRequested,dataType,format,nbDims,filterDimA)
    checkstatus(ccall((:cudnnGetFilterNdDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{cudnnTensorFormat_t},Ptr{Cint},Ptr{Cint}),filterDesc,nbDimsRequested,dataType,format,nbDims,filterDimA))
end

function cudnnDestroyFilterDescriptor(filterDesc)
    checkstatus(ccall((:cudnnDestroyFilterDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,),filterDesc))
end

function cudnnCreateConvolutionDescriptor(convDesc)
    checkstatus(ccall((:cudnnCreateConvolutionDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnConvolutionDescriptor_t},),convDesc))
end

function cudnnSetConvolution2dDescriptor(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode)
    checkstatus(ccall((:cudnnSetConvolution2dDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Cint,Cint,Cint,Cint,Cint,cudnnConvolutionMode_t),convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode))
end

function cudnnSetConvolution2dDescriptor_v5(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode,dataType)
    checkstatus(ccall((:cudnnSetConvolution2dDescriptor_v5,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Cint,Cint,Cint,Cint,Cint,cudnnConvolutionMode_t,cudnnDataType_t),convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode,dataType))
end

function cudnnGetConvolution2dDescriptor(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode)
    checkstatus(ccall((:cudnnGetConvolution2dDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t}),convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode))
end

function cudnnGetConvolution2dDescriptor_v5(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode,dataType)
    checkstatus(ccall((:cudnnGetConvolution2dDescriptor_v5,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t},Ptr{cudnnDataType_t}),convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode,dataType))
end

function cudnnGetConvolution2dForwardOutputDim(convDesc,inputTensorDesc,filterDesc,n,c,h,w)
    checkstatus(ccall((:cudnnGetConvolution2dForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),convDesc,inputTensorDesc,filterDesc,n,c,h,w))
end

function cudnnSetConvolutionNdDescriptor(convDesc,arrayLength,padA,filterStrideA,upscaleA,mode,dataType)
    checkstatus(ccall((:cudnnSetConvolutionNdDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},cudnnConvolutionMode_t,cudnnDataType_t),convDesc,arrayLength,padA,filterStrideA,upscaleA,mode,dataType))
end

function cudnnGetConvolutionNdDescriptor(convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode,dataType)
    checkstatus(ccall((:cudnnGetConvolutionNdDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t},Ptr{cudnnDataType_t}),convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode,dataType))
end

function cudnnGetConvolutionNdForwardOutputDim(convDesc,inputTensorDesc,filterDesc,nbDims,tensorOuputDimA)
    checkstatus(ccall((:cudnnGetConvolutionNdForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,Cint,Ptr{Cint}),convDesc,inputTensorDesc,filterDesc,nbDims,tensorOuputDimA))
end

function cudnnDestroyConvolutionDescriptor(convDesc)
    checkstatus(ccall((:cudnnDestroyConvolutionDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,),convDesc))
end

function cudnnFindConvolutionForwardAlgorithm(handle,xDesc,wDesc,convDesc,yDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
    checkstatus(ccall((:cudnnFindConvolutionForwardAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Cint},Ptr{cudnnConvolutionFwdAlgoPerf_t}),handle,xDesc,wDesc,convDesc,yDesc,requestedAlgoCount,returnedAlgoCount,perfResults))
end

function cudnnFindConvolutionForwardAlgorithmEx(handle,xDesc,x,wDesc,w,convDesc,yDesc,y,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes)
    checkstatus(ccall((:cudnnFindConvolutionForwardAlgorithmEx,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Ptr{Void},Cint,Ptr{Cint},Ptr{cudnnConvolutionFwdAlgoPerf_t},Ptr{Void},Csize_t),handle,xDesc,x,wDesc,w,convDesc,yDesc,y,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes))
end

function cudnnGetConvolutionForwardAlgorithm(handle,xDesc,wDesc,convDesc,yDesc,preference,memoryLimitInBytes,algo)
    checkstatus(ccall((:cudnnGetConvolutionForwardAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionFwdPreference_t,Csize_t,Ptr{cudnnConvolutionFwdAlgo_t}),handle,xDesc,wDesc,convDesc,yDesc,preference,memoryLimitInBytes,algo))
end

function cudnnGetConvolutionForwardWorkspaceSize(handle,xDesc,wDesc,convDesc,yDesc,algo,sizeInBytes)
    checkstatus(ccall((:cudnnGetConvolutionForwardWorkspaceSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionFwdAlgo_t,Ptr{Csize_t}),handle,xDesc,wDesc,convDesc,yDesc,algo,sizeInBytes))
end

function cudnnConvolutionForward(handle,alpha,xDesc,x,wDesc,w,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,yDesc,y)
    checkstatus(ccall((:cudnnConvolutionForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionFwdAlgo_t,Ptr{Void},Csize_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,xDesc,x,wDesc,w,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,yDesc,y))
end

function cudnnConvolutionBackwardBias(handle,alpha,dyDesc,dy,beta,dbDesc,db)
    checkstatus(ccall((:cudnnConvolutionBackwardBias,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,dyDesc,dy,beta,dbDesc,db))
end

function cudnnFindConvolutionBackwardFilterAlgorithm(handle,xDesc,dyDesc,convDesc,dwDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
    checkstatus(ccall((:cudnnFindConvolutionBackwardFilterAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnFilterDescriptor_t,Cint,Ptr{Cint},Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}),handle,xDesc,dyDesc,convDesc,dwDesc,requestedAlgoCount,returnedAlgoCount,perfResults))
end

function cudnnFindConvolutionBackwardFilterAlgorithmEx(handle,xDesc,x,dyDesc,y,convDesc,dwDesc,dw,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes)
    checkstatus(ccall((:cudnnFindConvolutionBackwardFilterAlgorithmEx,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnFilterDescriptor_t,Ptr{Void},Cint,Ptr{Cint},Ptr{cudnnConvolutionBwdFilterAlgoPerf_t},Ptr{Void},Csize_t),handle,xDesc,x,dyDesc,y,convDesc,dwDesc,dw,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes))
end

function cudnnGetConvolutionBackwardFilterAlgorithm(handle,xDesc,dyDesc,convDesc,dwDesc,preference,memoryLimitInBytes,algo)
    checkstatus(ccall((:cudnnGetConvolutionBackwardFilterAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionBwdFilterPreference_t,Csize_t,Ptr{cudnnConvolutionBwdFilterAlgo_t}),handle,xDesc,dyDesc,convDesc,dwDesc,preference,memoryLimitInBytes,algo))
end

function cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,xDesc,dyDesc,convDesc,gradDesc,algo,sizeInBytes)
    checkstatus(ccall((:cudnnGetConvolutionBackwardFilterWorkspaceSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionBwdFilterAlgo_t,Ptr{Csize_t}),handle,xDesc,dyDesc,convDesc,gradDesc,algo,sizeInBytes))
end

function cudnnConvolutionBackwardFilter(handle,alpha,xDesc,x,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dwDesc,dw)
    checkstatus(ccall((:cudnnConvolutionBackwardFilter,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionBwdFilterAlgo_t,Ptr{Void},Csize_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void}),handle,alpha,xDesc,x,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dwDesc,dw))
end

function cudnnFindConvolutionBackwardDataAlgorithm(handle,wDesc,dyDesc,convDesc,dxDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
    checkstatus(ccall((:cudnnFindConvolutionBackwardDataAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnFilterDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Cint},Ptr{cudnnConvolutionBwdDataAlgoPerf_t}),handle,wDesc,dyDesc,convDesc,dxDesc,requestedAlgoCount,returnedAlgoCount,perfResults))
end

function cudnnFindConvolutionBackwardDataAlgorithmEx(handle,wDesc,w,dyDesc,dy,convDesc,dxDesc,dx,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes)
    checkstatus(ccall((:cudnnFindConvolutionBackwardDataAlgorithmEx,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnFilterDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Ptr{Void},Cint,Ptr{Cint},Ptr{cudnnConvolutionBwdDataAlgoPerf_t},Ptr{Void},Csize_t),handle,wDesc,w,dyDesc,dy,convDesc,dxDesc,dx,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes))
end

function cudnnGetConvolutionBackwardDataAlgorithm(handle,wDesc,dyDesc,convDesc,dxDesc,preference,memoryLimitInBytes,algo)
    checkstatus(ccall((:cudnnGetConvolutionBackwardDataAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnFilterDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionBwdDataPreference_t,Csize_t,Ptr{cudnnConvolutionBwdDataAlgo_t}),handle,wDesc,dyDesc,convDesc,dxDesc,preference,memoryLimitInBytes,algo))
end

function cudnnGetConvolutionBackwardDataWorkspaceSize(handle,wDesc,dyDesc,convDesc,dxDesc,algo,sizeInBytes)
    checkstatus(ccall((:cudnnGetConvolutionBackwardDataWorkspaceSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnFilterDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionBwdDataAlgo_t,Ptr{Csize_t}),handle,wDesc,dyDesc,convDesc,dxDesc,algo,sizeInBytes))
end

function cudnnConvolutionBackwardData(handle,alpha,wDesc,w,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dxDesc,dx)
    checkstatus(ccall((:cudnnConvolutionBackwardData,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionBwdDataAlgo_t,Ptr{Void},Csize_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,wDesc,w,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dxDesc,dx))
end

function cudnnIm2Col(handle,xDesc,x,wDesc,convDesc,colBuffer)
    checkstatus(ccall((:cudnnIm2Col,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,Ptr{Void}),handle,xDesc,x,wDesc,convDesc,colBuffer))
end

function cudnnSoftmaxForward(handle,algo,mode,alpha,xDesc,x,beta,yDesc,y)
    checkstatus(ccall((:cudnnSoftmaxForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSoftmaxAlgorithm_t,cudnnSoftmaxMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algo,mode,alpha,xDesc,x,beta,yDesc,y))
end

function cudnnSoftmaxBackward(handle,algo,mode,alpha,yDesc,y,dyDesc,dy,beta,dxDesc,dx)
    checkstatus(ccall((:cudnnSoftmaxBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSoftmaxAlgorithm_t,cudnnSoftmaxMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algo,mode,alpha,yDesc,y,dyDesc,dy,beta,dxDesc,dx))
end

function cudnnCreatePoolingDescriptor(poolingDesc)
    checkstatus(ccall((:cudnnCreatePoolingDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnPoolingDescriptor_t},),poolingDesc))
end

function cudnnSetPooling2dDescriptor(poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
    checkstatus(ccall((:cudnnSetPooling2dDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,cudnnNanPropagation_t,Cint,Cint,Cint,Cint,Cint,Cint),poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride))
end

function cudnnGetPooling2dDescriptor(poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
    checkstatus(ccall((:cudnnGetPooling2dDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Ptr{cudnnPoolingMode_t},Ptr{cudnnNanPropagation_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride))
end

function cudnnSetPoolingNdDescriptor(poolingDesc,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA)
    checkstatus(ccall((:cudnnSetPoolingNdDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,cudnnNanPropagation_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA))
end

function cudnnGetPoolingNdDescriptor(poolingDesc,nbDimsRequested,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA)
    checkstatus(ccall((:cudnnGetPoolingNdDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Cint,Ptr{cudnnPoolingMode_t},Ptr{cudnnNanPropagation_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,nbDimsRequested,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA))
end

function cudnnGetPoolingNdForwardOutputDim(poolingDesc,inputTensorDesc,nbDims,outputTensorDimA)
    checkstatus(ccall((:cudnnGetPoolingNdForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Cint}),poolingDesc,inputTensorDesc,nbDims,outputTensorDimA))
end

function cudnnGetPooling2dForwardOutputDim(poolingDesc,inputTensorDesc,n,c,h,w)
    checkstatus(ccall((:cudnnGetPooling2dForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnTensorDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,inputTensorDesc,n,c,h,w))
end

function cudnnDestroyPoolingDescriptor(poolingDesc)
    checkstatus(ccall((:cudnnDestroyPoolingDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,),poolingDesc))
end

function cudnnPoolingForward(handle,poolingDesc,alpha,xDesc,x,beta,yDesc,y)
    checkstatus(ccall((:cudnnPoolingForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,poolingDesc,alpha,xDesc,x,beta,yDesc,y))
end

function cudnnPoolingBackward(handle,poolingDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
    checkstatus(ccall((:cudnnPoolingBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,poolingDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx))
end

function cudnnCreateActivationDescriptor(activationDesc)
    checkstatus(ccall((:cudnnCreateActivationDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnActivationDescriptor_t},),activationDesc))
end

function cudnnSetActivationDescriptor(activationDesc,mode,reluNanOpt,reluCeiling)
    checkstatus(ccall((:cudnnSetActivationDescriptor,libcudnn),cudnnStatus_t,(cudnnActivationDescriptor_t,cudnnActivationMode_t,cudnnNanPropagation_t,Cdouble),activationDesc,mode,reluNanOpt,reluCeiling))
end

function cudnnGetActivationDescriptor(activationDesc,mode,reluNanOpt,reluCeiling)
    checkstatus(ccall((:cudnnGetActivationDescriptor,libcudnn),cudnnStatus_t,(cudnnActivationDescriptor_t,Ptr{cudnnActivationMode_t},Ptr{cudnnNanPropagation_t},Ptr{Cdouble}),activationDesc,mode,reluNanOpt,reluCeiling))
end

function cudnnDestroyActivationDescriptor(activationDesc)
    checkstatus(ccall((:cudnnDestroyActivationDescriptor,libcudnn),cudnnStatus_t,(cudnnActivationDescriptor_t,),activationDesc))
end

function cudnnActivationForward(handle,activationDesc,alpha,xDesc,x,beta,yDesc,y)
    checkstatus(ccall((:cudnnActivationForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,activationDesc,alpha,xDesc,x,beta,yDesc,y))
end

function cudnnActivationBackward(handle,activationDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
    checkstatus(ccall((:cudnnActivationBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,activationDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx))
end

function cudnnCreateLRNDescriptor(normDesc)
    checkstatus(ccall((:cudnnCreateLRNDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnLRNDescriptor_t},),normDesc))
end

function cudnnSetLRNDescriptor(normDesc,lrnN,lrnAlpha,lrnBeta,lrnK)
    checkstatus(ccall((:cudnnSetLRNDescriptor,libcudnn),cudnnStatus_t,(cudnnLRNDescriptor_t,UInt32,Cdouble,Cdouble,Cdouble),normDesc,lrnN,lrnAlpha,lrnBeta,lrnK))
end

function cudnnGetLRNDescriptor(normDesc,lrnN,lrnAlpha,lrnBeta,lrnK)
    checkstatus(ccall((:cudnnGetLRNDescriptor,libcudnn),cudnnStatus_t,(cudnnLRNDescriptor_t,Ptr{UInt32},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),normDesc,lrnN,lrnAlpha,lrnBeta,lrnK))
end

function cudnnDestroyLRNDescriptor(lrnDesc)
    checkstatus(ccall((:cudnnDestroyLRNDescriptor,libcudnn),cudnnStatus_t,(cudnnLRNDescriptor_t,),lrnDesc))
end

function cudnnLRNCrossChannelForward(handle,normDesc,lrnMode,alpha,xDesc,x,beta,yDesc,y)
    checkstatus(ccall((:cudnnLRNCrossChannelForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnLRNDescriptor_t,cudnnLRNMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,normDesc,lrnMode,alpha,xDesc,x,beta,yDesc,y))
end

function cudnnLRNCrossChannelBackward(handle,normDesc,lrnMode,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
    checkstatus(ccall((:cudnnLRNCrossChannelBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnLRNDescriptor_t,cudnnLRNMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,normDesc,lrnMode,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx))
end

function cudnnDivisiveNormalizationForward(handle,normDesc,mode,alpha,xDesc,x,means,temp,temp2,beta,yDesc,y)
    checkstatus(ccall((:cudnnDivisiveNormalizationForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnLRNDescriptor_t,cudnnDivNormMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,normDesc,mode,alpha,xDesc,x,means,temp,temp2,beta,yDesc,y))
end

function cudnnDivisiveNormalizationBackward(handle,normDesc,mode,alpha,xDesc,x,means,dy,temp,temp2,beta,dXdMeansDesc,dx,dMeans)
    checkstatus(ccall((:cudnnDivisiveNormalizationBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnLRNDescriptor_t,cudnnDivNormMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,normDesc,mode,alpha,xDesc,x,means,dy,temp,temp2,beta,dXdMeansDesc,dx,dMeans))
end

function cudnnDeriveBNTensorDescriptor(derivedBnDesc,xDesc,mode)
    checkstatus(ccall((:cudnnDeriveBNTensorDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnBatchNormMode_t),derivedBnDesc,xDesc,mode))
end

function cudnnBatchNormalizationForwardTraining(handle,mode,alpha,beta,xDesc,x,yDesc,y,bnScaleBiasMeanVarDesc,bnScale,bnBias,exponentialAverageFactor,resultRunningMean,resultRunningVariance,epsilon,resultSaveMean,resultSaveInvVariance)
    checkstatus(ccall((:cudnnBatchNormalizationForwardTraining,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnBatchNormMode_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Cdouble,Ptr{Void},Ptr{Void},Cdouble,Ptr{Void},Ptr{Void}),handle,mode,alpha,beta,xDesc,x,yDesc,y,bnScaleBiasMeanVarDesc,bnScale,bnBias,exponentialAverageFactor,resultRunningMean,resultRunningVariance,epsilon,resultSaveMean,resultSaveInvVariance))
end

function cudnnBatchNormalizationForwardInference(handle,mode,alpha,beta,xDesc,x,yDesc,y,bnScaleBiasMeanVarDesc,bnScale,bnBias,estimatedMean,estimatedVariance,epsilon)
    checkstatus(ccall((:cudnnBatchNormalizationForwardInference,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnBatchNormMode_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Cdouble),handle,mode,alpha,beta,xDesc,x,yDesc,y,bnScaleBiasMeanVarDesc,bnScale,bnBias,estimatedMean,estimatedVariance,epsilon))
end

function cudnnBatchNormalizationBackward(handle,mode,alphaDataDiff,betaDataDiff,alphaParamDiff,betaParamDiff,xDesc,x,dyDesc,dy,dxDesc,dx,dBnScaleBiasDesc,bnScale,dBnScaleResult,dBnBiasResult,epsilon,savedMean,savedInvVariance)
    checkstatus(ccall((:cudnnBatchNormalizationBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnBatchNormMode_t,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},Cdouble,Ptr{Void},Ptr{Void}),handle,mode,alphaDataDiff,betaDataDiff,alphaParamDiff,betaParamDiff,xDesc,x,dyDesc,dy,dxDesc,dx,dBnScaleBiasDesc,bnScale,dBnScaleResult,dBnBiasResult,epsilon,savedMean,savedInvVariance))
end

function cudnnCreateSpatialTransformerDescriptor(stDesc)
    checkstatus(ccall((:cudnnCreateSpatialTransformerDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnSpatialTransformerDescriptor_t},),stDesc))
end

function cudnnSetSpatialTransformerNdDescriptor(stDesc,samplerType,dataType,nbDims,dimA)
    checkstatus(ccall((:cudnnSetSpatialTransformerNdDescriptor,libcudnn),cudnnStatus_t,(cudnnSpatialTransformerDescriptor_t,cudnnSamplerType_t,cudnnDataType_t,Cint,Ptr{Cint}),stDesc,samplerType,dataType,nbDims,dimA))
end

function cudnnDestroySpatialTransformerDescriptor(stDesc)
    checkstatus(ccall((:cudnnDestroySpatialTransformerDescriptor,libcudnn),cudnnStatus_t,(cudnnSpatialTransformerDescriptor_t,),stDesc))
end

function cudnnSpatialTfGridGeneratorForward(handle,stDesc,theta,grid)
    checkstatus(ccall((:cudnnSpatialTfGridGeneratorForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSpatialTransformerDescriptor_t,Ptr{Void},Ptr{Void}),handle,stDesc,theta,grid))
end

function cudnnSpatialTfGridGeneratorBackward(handle,stDesc,dgrid,dtheta)
    checkstatus(ccall((:cudnnSpatialTfGridGeneratorBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSpatialTransformerDescriptor_t,Ptr{Void},Ptr{Void}),handle,stDesc,dgrid,dtheta))
end

function cudnnSpatialTfSamplerForward(handle,stDesc,alpha,xDesc,x,grid,beta,yDesc,y)
    checkstatus(ccall((:cudnnSpatialTfSamplerForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSpatialTransformerDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,stDesc,alpha,xDesc,x,grid,beta,yDesc,y))
end

function cudnnSpatialTfSamplerBackward(handle,stDesc,alpha,xDesc,x,beta,dxDesc,dx,alphaDgrid,dyDesc,dy,grid,betaDgrid,dgrid)
    checkstatus(ccall((:cudnnSpatialTfSamplerBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSpatialTransformerDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),handle,stDesc,alpha,xDesc,x,beta,dxDesc,dx,alphaDgrid,dyDesc,dy,grid,betaDgrid,dgrid))
end

function cudnnCreateDropoutDescriptor(dropoutDesc)
    checkstatus(ccall((:cudnnCreateDropoutDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnDropoutDescriptor_t},),dropoutDesc))
end

function cudnnDestroyDropoutDescriptor(dropoutDesc)
    checkstatus(ccall((:cudnnDestroyDropoutDescriptor,libcudnn),cudnnStatus_t,(cudnnDropoutDescriptor_t,),dropoutDesc))
end

function cudnnDropoutGetStatesSize(handle,sizeInBytes)
    checkstatus(ccall((:cudnnDropoutGetStatesSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Csize_t}),handle,sizeInBytes))
end

function cudnnDropoutGetReserveSpaceSize(xdesc,sizeInBytes)
    checkstatus(ccall((:cudnnDropoutGetReserveSpaceSize,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,Ptr{Csize_t}),xdesc,sizeInBytes))
end

function cudnnSetDropoutDescriptor(dropoutDesc,handle,dropout,states,stateSizeInBytes,seed)
    checkstatus(ccall((:cudnnSetDropoutDescriptor,libcudnn),cudnnStatus_t,(cudnnDropoutDescriptor_t,cudnnHandle_t,Cfloat,Ptr{Void},Csize_t,Culonglong),dropoutDesc,handle,dropout,states,stateSizeInBytes,seed))
end

function cudnnDropoutForward(handle,dropoutDesc,xdesc,x,ydesc,y,reserveSpace,reserveSpaceSizeInBytes)
    checkstatus(ccall((:cudnnDropoutForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnDropoutDescriptor_t,cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Csize_t),handle,dropoutDesc,xdesc,x,ydesc,y,reserveSpace,reserveSpaceSizeInBytes))
end

function cudnnDropoutBackward(handle,dropoutDesc,dydesc,dy,dxdesc,dx,reserveSpace,reserveSpaceSizeInBytes)
    checkstatus(ccall((:cudnnDropoutBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnDropoutDescriptor_t,cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Csize_t),handle,dropoutDesc,dydesc,dy,dxdesc,dx,reserveSpace,reserveSpaceSizeInBytes))
end

function cudnnCreateRNNDescriptor(rnnDesc)
    checkstatus(ccall((:cudnnCreateRNNDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnRNNDescriptor_t},),rnnDesc))
end

function cudnnDestroyRNNDescriptor(rnnDesc)
    checkstatus(ccall((:cudnnDestroyRNNDescriptor,libcudnn),cudnnStatus_t,(cudnnRNNDescriptor_t,),rnnDesc))
end

function cudnnSetRNNDescriptor(rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,dataType)
    checkstatus(ccall((:cudnnSetRNNDescriptor,libcudnn),cudnnStatus_t,(cudnnRNNDescriptor_t,Cint,Cint,cudnnDropoutDescriptor_t,cudnnRNNInputMode_t,cudnnDirectionMode_t,cudnnRNNMode_t,cudnnDataType_t),rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,dataType))
end

function cudnnGetRNNWorkspaceSize(handle,rnnDesc,seqLength,xDesc,sizeInBytes)
    checkstatus(ccall((:cudnnGetRNNWorkspaceSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnRNNDescriptor_t,Cint,Ptr{cudnnTensorDescriptor_t},Ptr{Csize_t}),handle,rnnDesc,seqLength,xDesc,sizeInBytes))
end

function cudnnGetRNNTrainingReserveSize(handle,rnnDesc,seqLength,xDesc,sizeInBytes)
    checkstatus(ccall((:cudnnGetRNNTrainingReserveSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnRNNDescriptor_t,Cint,Ptr{cudnnTensorDescriptor_t},Ptr{Csize_t}),handle,rnnDesc,seqLength,xDesc,sizeInBytes))
end

function cudnnGetRNNParamsSize(handle,rnnDesc,xDesc,sizeInBytes,dataType)
    checkstatus(ccall((:cudnnGetRNNParamsSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnRNNDescriptor_t,cudnnTensorDescriptor_t,Ptr{Csize_t},cudnnDataType_t),handle,rnnDesc,xDesc,sizeInBytes,dataType))
end

function cudnnGetRNNLinLayerMatrixParams(handle,rnnDesc,layer,xDesc,wDesc,w,linLayerID,linLayerMatDesc,linLayerMat)
    checkstatus(ccall((:cudnnGetRNNLinLayerMatrixParams,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnRNNDescriptor_t,Cint,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,Ptr{Void},Cint,cudnnFilterDescriptor_t,Ptr{Ptr{Void}}),handle,rnnDesc,layer,xDesc,wDesc,w,linLayerID,linLayerMatDesc,linLayerMat))
end

function cudnnGetRNNLinLayerBiasParams(handle,rnnDesc,layer,xDesc,wDesc,w,linLayerID,linLayerBiasDesc,linLayerBias)
    checkstatus(ccall((:cudnnGetRNNLinLayerBiasParams,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnRNNDescriptor_t,Cint,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,Ptr{Void},Cint,cudnnFilterDescriptor_t,Ptr{Ptr{Void}}),handle,rnnDesc,layer,xDesc,wDesc,w,linLayerID,linLayerBiasDesc,linLayerBias))
end

function cudnnRNNForwardInference(handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,cxDesc,cx,wDesc,w,yDesc,y,hyDesc,hy,cyDesc,cy,workspace,workSpaceSizeInBytes)
    checkstatus(ccall((:cudnnRNNForwardInference,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnRNNDescriptor_t,Cint,Ptr{cudnnTensorDescriptor_t},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},Ptr{cudnnTensorDescriptor_t},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Csize_t),handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,cxDesc,cx,wDesc,w,yDesc,y,hyDesc,hy,cyDesc,cy,workspace,workSpaceSizeInBytes))
end

function cudnnRNNForwardTraining(handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,cxDesc,cx,wDesc,w,yDesc,y,hyDesc,hy,cyDesc,cy,workspace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes)
    checkstatus(ccall((:cudnnRNNForwardTraining,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnRNNDescriptor_t,Cint,Ptr{cudnnTensorDescriptor_t},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},Ptr{cudnnTensorDescriptor_t},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Csize_t,Ptr{Void},Csize_t),handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,cxDesc,cx,wDesc,w,yDesc,y,hyDesc,hy,cyDesc,cy,workspace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes))
end

function cudnnRNNBackwardData(handle,rnnDesc,seqLength,yDesc,y,dyDesc,dy,dhyDesc,dhy,dcyDesc,dcy,wDesc,w,hxDesc,hx,cxDesc,cx,dxDesc,dx,dhxDesc,dhx,dcxDesc,dcx,workspace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes)
    checkstatus(ccall((:cudnnRNNBackwardData,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnRNNDescriptor_t,Cint,Ptr{cudnnTensorDescriptor_t},Ptr{Void},Ptr{cudnnTensorDescriptor_t},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{cudnnTensorDescriptor_t},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Csize_t,Ptr{Void},Csize_t),handle,rnnDesc,seqLength,yDesc,y,dyDesc,dy,dhyDesc,dhy,dcyDesc,dcy,wDesc,w,hxDesc,hx,cxDesc,cx,dxDesc,dx,dhxDesc,dhx,dcxDesc,dcx,workspace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes))
end

function cudnnRNNBackwardWeights(handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,yDesc,y,workspace,workSpaceSizeInBytes,dwDesc,dw,reserveSpace,reserveSpaceSizeInBytes)
    checkstatus(ccall((:cudnnRNNBackwardWeights,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnRNNDescriptor_t,Cint,Ptr{cudnnTensorDescriptor_t},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{cudnnTensorDescriptor_t},Ptr{Void},Ptr{Void},Csize_t,cudnnFilterDescriptor_t,Ptr{Void},Ptr{Void},Csize_t),handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,yDesc,y,workspace,workSpaceSizeInBytes,dwDesc,dw,reserveSpace,reserveSpaceSizeInBytes))
end

function cudnnSetFilter4dDescriptor_v3(filterDesc,dataType,k,c,h,w)
    checkstatus(ccall((:cudnnSetFilter4dDescriptor_v3,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,Cint,Cint,Cint,Cint),filterDesc,dataType,k,c,h,w))
end

function cudnnSetFilter4dDescriptor_v4(filterDesc,dataType,format,k,c,h,w)
    checkstatus(ccall((:cudnnSetFilter4dDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,cudnnTensorFormat_t,Cint,Cint,Cint,Cint),filterDesc,dataType,format,k,c,h,w))
end

function cudnnGetFilter4dDescriptor_v3(filterDesc,dataType,k,c,h,w)
    checkstatus(ccall((:cudnnGetFilter4dDescriptor_v3,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),filterDesc,dataType,k,c,h,w))
end

function cudnnGetFilter4dDescriptor_v4(filterDesc,dataType,format,k,c,h,w)
    checkstatus(ccall((:cudnnGetFilter4dDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Ptr{cudnnDataType_t},Ptr{cudnnTensorFormat_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),filterDesc,dataType,format,k,c,h,w))
end

function cudnnSetFilterNdDescriptor_v3(filterDesc,dataType,nbDims,filterDimA)
    checkstatus(ccall((:cudnnSetFilterNdDescriptor_v3,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,Cint,Ptr{Cint}),filterDesc,dataType,nbDims,filterDimA))
end

function cudnnSetFilterNdDescriptor_v4(filterDesc,dataType,format,nbDims,filterDimA)
    checkstatus(ccall((:cudnnSetFilterNdDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,cudnnTensorFormat_t,Cint,Ptr{Cint}),filterDesc,dataType,format,nbDims,filterDimA))
end

function cudnnGetFilterNdDescriptor_v3(filterDesc,nbDimsRequested,dataType,nbDims,filterDimA)
    checkstatus(ccall((:cudnnGetFilterNdDescriptor_v3,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint}),filterDesc,nbDimsRequested,dataType,nbDims,filterDimA))
end

function cudnnGetFilterNdDescriptor_v4(filterDesc,nbDimsRequested,dataType,format,nbDims,filterDimA)
    checkstatus(ccall((:cudnnGetFilterNdDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{cudnnTensorFormat_t},Ptr{Cint},Ptr{Cint}),filterDesc,nbDimsRequested,dataType,format,nbDims,filterDimA))
end

function cudnnSetPooling2dDescriptor_v3(poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
    checkstatus(ccall((:cudnnSetPooling2dDescriptor_v3,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,Cint,Cint,Cint,Cint,Cint,Cint),poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride))
end

function cudnnSetPooling2dDescriptor_v4(poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
    checkstatus(ccall((:cudnnSetPooling2dDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,cudnnNanPropagation_t,Cint,Cint,Cint,Cint,Cint,Cint),poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride))
end

function cudnnGetPooling2dDescriptor_v3(poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
    checkstatus(ccall((:cudnnGetPooling2dDescriptor_v3,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Ptr{cudnnPoolingMode_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride))
end

function cudnnGetPooling2dDescriptor_v4(poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
    checkstatus(ccall((:cudnnGetPooling2dDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Ptr{cudnnPoolingMode_t},Ptr{cudnnNanPropagation_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride))
end

function cudnnSetPoolingNdDescriptor_v3(poolingDesc,mode,nbDims,windowDimA,paddingA,strideA)
    checkstatus(ccall((:cudnnSetPoolingNdDescriptor_v3,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,nbDims,windowDimA,paddingA,strideA))
end

function cudnnSetPoolingNdDescriptor_v4(poolingDesc,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA)
    checkstatus(ccall((:cudnnSetPoolingNdDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,cudnnNanPropagation_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA))
end

function cudnnGetPoolingNdDescriptor_v3(poolingDesc,nbDimsRequested,mode,nbDims,windowDimA,paddingA,strideA)
    checkstatus(ccall((:cudnnGetPoolingNdDescriptor_v3,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Cint,Ptr{cudnnPoolingMode_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,nbDimsRequested,mode,nbDims,windowDimA,paddingA,strideA))
end

function cudnnGetPoolingNdDescriptor_v4(poolingDesc,nbDimsRequested,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA)
    checkstatus(ccall((:cudnnGetPoolingNdDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Cint,Ptr{cudnnPoolingMode_t},Ptr{cudnnNanPropagation_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,nbDimsRequested,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA))
end

function cudnnActivationForward_v3(handle,mode,alpha,xDesc,x,beta,yDesc,y)
    checkstatus(ccall((:cudnnActivationForward_v3,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,xDesc,x,beta,yDesc,y))
end

function cudnnActivationForward_v4(handle,activationDesc,alpha,xDesc,x,beta,yDesc,y)
    checkstatus(ccall((:cudnnActivationForward_v4,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,activationDesc,alpha,xDesc,x,beta,yDesc,y))
end

function cudnnActivationBackward_v3(handle,mode,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
    checkstatus(ccall((:cudnnActivationBackward_v3,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx))
end

function cudnnActivationBackward_v4(handle,activationDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
    checkstatus(ccall((:cudnnActivationBackward_v4,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,activationDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx))
end
