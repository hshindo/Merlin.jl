# Julia wrapper for header: /home/cl/shindo/lime/cudnn.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function cudnnGetVersion()
    ccall((:cudnnGetVersion, libcudnn), Csize_t, ())
end

function cudnnGetCudartVersion()
    ccall((:cudnnGetCudartVersion, libcudnn), Csize_t, ())
end

function cudnnGetErrorString(status)
    ccall((:cudnnGetErrorString, libcudnn), Cstring, (cudnnStatus_t,), status)
end

function cudnnQueryRuntimeError(handle, rstatus, mode, tag)
    ccall((:cudnnQueryRuntimeError, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{cudnnStatus_t}, cudnnErrQueryMode_t, Ptr{cudnnRuntimeTag_t}), handle, rstatus, mode, tag)
end

function cudnnGetProperty(_type, value)
    ccall((:cudnnGetProperty, libcudnn), cudnnStatus_t, (libraryPropertyType, Ptr{Cint}), _type, value)
end

function cudnnCreate(handle)
    ccall((:cudnnCreate, libcudnn), cudnnStatus_t, (Ptr{cudnnHandle_t},), handle)
end

function cudnnDestroy(handle)
    ccall((:cudnnDestroy, libcudnn), cudnnStatus_t, (cudnnHandle_t,), handle)
end

function cudnnSetStream(handle, streamId)
    ccall((:cudnnSetStream, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudaStream_t), handle, streamId)
end

function cudnnGetStream(handle, streamId)
    ccall((:cudnnGetStream, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{cudaStream_t}), handle, streamId)
end

function cudnnCreateTensorDescriptor(tensorDesc)
    ccall((:cudnnCreateTensorDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnTensorDescriptor_t},), tensorDesc)
end

function cudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w)
    ccall((:cudnnSetTensor4dDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, Cint, Cint, Cint, Cint), tensorDesc, format, dataType, n, c, h, w)
end

function cudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
    ccall((:cudnnSetTensor4dDescriptorEx, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnDataType_t, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint), tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
end

function cudnnGetTensor4dDescriptor(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
    ccall((:cudnnGetTensor4dDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, Ptr{cudnnDataType_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
end

function cudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA)
    ccall((:cudnnSetTensorNdDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnDataType_t, Cint, Ptr{Cint}, Ptr{Cint}), tensorDesc, dataType, nbDims, dimA, strideA)
end

function cudnnSetTensorNdDescriptorEx(tensorDesc, format, dataType, nbDims, dimA)
    ccall((:cudnnSetTensorNdDescriptorEx, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, Cint, Ptr{Cint}), tensorDesc, format, dataType, nbDims, dimA)
end

function cudnnGetTensorNdDescriptor(tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA)
    ccall((:cudnnGetTensorNdDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, Cint, Ptr{cudnnDataType_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA)
end

function cudnnGetTensorSizeInBytes(tensorDesc, size)
    ccall((:cudnnGetTensorSizeInBytes, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, Ptr{Csize_t}), tensorDesc, size)
end

function cudnnDestroyTensorDescriptor(tensorDesc)
    ccall((:cudnnDestroyTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t,), tensorDesc)
end

function cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y)
    ccall((:cudnnTransformTensor, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C)
    ccall((:cudnnAddTensor, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, alpha, aDesc, A, beta, cDesc, C)
end

function cudnnCreateOpTensorDescriptor(opTensorDesc)
    ccall((:cudnnCreateOpTensorDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnOpTensorDescriptor_t},), opTensorDesc)
end

function cudnnSetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
    ccall((:cudnnSetOpTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t, cudnnDataType_t, cudnnNanPropagation_t), opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
end

function cudnnGetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
    ccall((:cudnnGetOpTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnOpTensorDescriptor_t, Ptr{cudnnOpTensorOp_t}, Ptr{cudnnDataType_t}, Ptr{cudnnNanPropagation_t}), opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
end

function cudnnDestroyOpTensorDescriptor(opTensorDesc)
    ccall((:cudnnDestroyOpTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnOpTensorDescriptor_t,), opTensorDesc)
end

function cudnnOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C)
    ccall((:cudnnOpTensor, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnOpTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C)
end

function cudnnCreateReduceTensorDescriptor(reduceTensorDesc)
    ccall((:cudnnCreateReduceTensorDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnReduceTensorDescriptor_t},), reduceTensorDesc)
end

function cudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType)
    ccall((:cudnnSetReduceTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnReduceTensorDescriptor_t, cudnnReduceTensorOp_t, cudnnDataType_t, cudnnNanPropagation_t, cudnnReduceTensorIndices_t, cudnnIndicesType_t), reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType)
end

function cudnnGetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType)
    ccall((:cudnnGetReduceTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnReduceTensorDescriptor_t, Ptr{cudnnReduceTensorOp_t}, Ptr{cudnnDataType_t}, Ptr{cudnnNanPropagation_t}, Ptr{cudnnReduceTensorIndices_t}, Ptr{cudnnIndicesType_t}), reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType)
end

function cudnnDestroyReduceTensorDescriptor(reduceTensorDesc)
    ccall((:cudnnDestroyReduceTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnReduceTensorDescriptor_t,), reduceTensorDesc)
end

function cudnnGetReductionIndicesSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
    ccall((:cudnnGetReductionIndicesSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnReduceTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, Ptr{Csize_t}), handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
end

function cudnnGetReductionWorkspaceSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
    ccall((:cudnnGetReductionWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnReduceTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, Ptr{Csize_t}), handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
end

function cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C)
    ccall((:cudnnReduceTensor, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnReduceTensorDescriptor_t, Ptr{Void}, Csize_t, Ptr{Void}, Csize_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C)
end

function cudnnSetTensor(handle, yDesc, y, valuePtr)
    ccall((:cudnnSetTensor, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}), handle, yDesc, y, valuePtr)
end

function cudnnScaleTensor(handle, yDesc, y, alpha)
    ccall((:cudnnScaleTensor, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}), handle, yDesc, y, alpha)
end

function cudnnCreateFilterDescriptor(filterDesc)
    ccall((:cudnnCreateFilterDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnFilterDescriptor_t},), filterDesc)
end

function cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w)
    ccall((:cudnnSetFilter4dDescriptor, libcudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, Cint, Cint, Cint, Cint), filterDesc, dataType, format, k, c, h, w)
end

function cudnnGetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w)
    ccall((:cudnnGetFilter4dDescriptor, libcudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, Ptr{cudnnDataType_t}, Ptr{cudnnTensorFormat_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), filterDesc, dataType, format, k, c, h, w)
end

function cudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims, filterDimA)
    ccall((:cudnnSetFilterNdDescriptor, libcudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, Cint, Ptr{Cint}), filterDesc, dataType, format, nbDims, filterDimA)
end

function cudnnGetFilterNdDescriptor(filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)
    ccall((:cudnnGetFilterNdDescriptor, libcudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, Cint, Ptr{cudnnDataType_t}, Ptr{cudnnTensorFormat_t}, Ptr{Cint}, Ptr{Cint}), filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)
end

function cudnnDestroyFilterDescriptor(filterDesc)
    ccall((:cudnnDestroyFilterDescriptor, libcudnn), cudnnStatus_t, (cudnnFilterDescriptor_t,), filterDesc)
end

function cudnnCreateConvolutionDescriptor(convDesc)
    ccall((:cudnnCreateConvolutionDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnConvolutionDescriptor_t},), convDesc)
end

function cudnnSetConvolutionMathType(convDesc, mathType)
    ccall((:cudnnSetConvolutionMathType, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, cudnnMathType_t), convDesc, mathType)
end

function cudnnGetConvolutionMathType(convDesc, mathType)
    ccall((:cudnnGetConvolutionMathType, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Ptr{cudnnMathType_t}), convDesc, mathType)
end

function cudnnSetConvolutionGroupCount(convDesc, groupCount)
    ccall((:cudnnSetConvolutionGroupCount, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Cint), convDesc, groupCount)
end

function cudnnGetConvolutionGroupCount(convDesc, groupCount)
    ccall((:cudnnGetConvolutionGroupCount, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Ptr{Cint}), convDesc, groupCount)
end

function cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
    ccall((:cudnnSetConvolution2dDescriptor, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Cint, Cint, Cint, Cint, Cint, Cint, cudnnConvolutionMode_t, cudnnDataType_t), convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
end

function cudnnGetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
    ccall((:cudnnGetConvolution2dDescriptor, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{cudnnConvolutionMode_t}, Ptr{cudnnDataType_t}), convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
end

function cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensorDesc, filterDesc, n, c, h, w)
    ccall((:cudnnGetConvolution2dForwardOutputDim, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), convDesc, inputTensorDesc, filterDesc, n, c, h, w)
end

function cudnnSetConvolutionNdDescriptor(convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType)
    ccall((:cudnnSetConvolutionNdDescriptor, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, cudnnConvolutionMode_t, cudnnDataType_t), convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType)
end

function cudnnGetConvolutionNdDescriptor(convDesc, arrayLengthRequested, arrayLength, padA, strideA, dilationA, mode, computeType)
    ccall((:cudnnGetConvolutionNdDescriptor, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{cudnnConvolutionMode_t}, Ptr{cudnnDataType_t}), convDesc, arrayLengthRequested, arrayLength, padA, strideA, dilationA, mode, computeType)
end

function cudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA)
    ccall((:cudnnGetConvolutionNdForwardOutputDim, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, Cint, Ptr{Cint}), convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA)
end

function cudnnDestroyConvolutionDescriptor(convDesc)
    ccall((:cudnnDestroyConvolutionDescriptor, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t,), convDesc)
end

function cudnnGetConvolutionForwardAlgorithmMaxCount(handle, count)
    ccall((:cudnnGetConvolutionForwardAlgorithmMaxCount, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cint}), handle, count)
end

function cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    ccall((:cudnnFindConvolutionForwardAlgorithm, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionFwdAlgoPerf_t}), handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
end

function cudnnFindConvolutionForwardAlgorithmEx(handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
    ccall((:cudnnFindConvolutionForwardAlgorithmEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Ptr{Void}, Cint, Ptr{Cint}, Ptr{cudnnConvolutionFwdAlgoPerf_t}, Ptr{Void}, Csize_t), handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
end

function cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, algo)
    ccall((:cudnnGetConvolutionForwardAlgorithm, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionFwdPreference_t, Csize_t, Ptr{cudnnConvolutionFwdAlgo_t}), handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, algo)
end

function cudnnGetConvolutionForwardAlgorithm_v7(handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    ccall((:cudnnGetConvolutionForwardAlgorithm_v7, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionFwdAlgoPerf_t}), handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
end

function cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes)
    ccall((:cudnnGetConvolutionForwardWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionFwdAlgo_t, Ptr{Csize_t}), handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes)
end

function cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y)
    ccall((:cudnnConvolutionForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}, cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, Ptr{Void}, Csize_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y)
end

function cudnnConvolutionBiasActivationForward(handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc, y)
    ccall((:cudnnConvolutionBiasActivationForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}, cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, Ptr{Void}, Csize_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, Ptr{Void}), handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc, y)
end

function cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db)
    ccall((:cudnnConvolutionBackwardBias, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, alpha, dyDesc, dy, beta, dbDesc, db)
end

function cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, count)
    ccall((:cudnnGetConvolutionBackwardFilterAlgorithmMaxCount, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cint}), handle, count)
end

function cudnnFindConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    ccall((:cudnnFindConvolutionBackwardFilterAlgorithm, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}), handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
end

function cudnnFindConvolutionBackwardFilterAlgorithmEx(handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
    ccall((:cudnnFindConvolutionBackwardFilterAlgorithmEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, Ptr{Void}, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}, Ptr{Void}, Csize_t), handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
end

function cudnnGetConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, algo)
    ccall((:cudnnGetConvolutionBackwardFilterAlgorithm, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionBwdFilterPreference_t, Csize_t, Ptr{cudnnConvolutionBwdFilterAlgo_t}), handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, algo)
end

function cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    ccall((:cudnnGetConvolutionBackwardFilterAlgorithm_v7, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}), handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
end

function cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes)
    ccall((:cudnnGetConvolutionBackwardFilterWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionBwdFilterAlgo_t, Ptr{Csize_t}), handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes)
end

function cudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw)
    ccall((:cudnnConvolutionBackwardFilter, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdFilterAlgo_t, Ptr{Void}, Csize_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}), handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw)
end

function cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, count)
    ccall((:cudnnGetConvolutionBackwardDataAlgorithmMaxCount, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cint}), handle, count)
end

function cudnnFindConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    ccall((:cudnnFindConvolutionBackwardDataAlgorithm, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdDataAlgoPerf_t}), handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
end

function cudnnFindConvolutionBackwardDataAlgorithmEx(handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
    ccall((:cudnnFindConvolutionBackwardDataAlgorithmEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Ptr{Void}, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdDataAlgoPerf_t}, Ptr{Void}, Csize_t), handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
end

function cudnnGetConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, algo)
    ccall((:cudnnGetConvolutionBackwardDataAlgorithm, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionBwdDataPreference_t, Csize_t, Ptr{cudnnConvolutionBwdDataAlgo_t}), handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, algo)
end

function cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    ccall((:cudnnGetConvolutionBackwardDataAlgorithm_v7, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdDataAlgoPerf_t}), handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
end

function cudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes)
    ccall((:cudnnGetConvolutionBackwardDataWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionBwdDataAlgo_t, Ptr{Csize_t}), handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes)
end

function cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx)
    ccall((:cudnnConvolutionBackwardData, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdDataAlgo_t, Ptr{Void}, Csize_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx)
end

function cudnnIm2Col(handle, xDesc, x, wDesc, convDesc, colBuffer)
    ccall((:cudnnIm2Col, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, Ptr{Void}, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, Ptr{Void}), handle, xDesc, x, wDesc, convDesc, colBuffer)
end

function cudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y)
    ccall((:cudnnSoftmaxForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, algo, mode, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnSoftmaxBackward(handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx)
    ccall((:cudnnSoftmaxBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx)
end

function cudnnCreatePoolingDescriptor(poolingDesc)
    ccall((:cudnnCreatePoolingDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnPoolingDescriptor_t},), poolingDesc)
end

function cudnnSetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
    ccall((:cudnnSetPooling2dDescriptor, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, Cint, Cint, Cint, Cint, Cint, Cint), poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
end

function cudnnGetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
    ccall((:cudnnGetPooling2dDescriptor, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, Ptr{cudnnPoolingMode_t}, Ptr{cudnnNanPropagation_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
end

function cudnnSetPoolingNdDescriptor(poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
    ccall((:cudnnSetPoolingNdDescriptor, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
end

function cudnnGetPoolingNdDescriptor(poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
    ccall((:cudnnGetPoolingNdDescriptor, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, Cint, Ptr{cudnnPoolingMode_t}, Ptr{cudnnNanPropagation_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
end

function cudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA)
    ccall((:cudnnGetPoolingNdForwardOutputDim, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}), poolingDesc, inputTensorDesc, nbDims, outputTensorDimA)
end

function cudnnGetPooling2dForwardOutputDim(poolingDesc, inputTensorDesc, n, c, h, w)
    ccall((:cudnnGetPooling2dForwardOutputDim, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnTensorDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, inputTensorDesc, n, c, h, w)
end

function cudnnDestroyPoolingDescriptor(poolingDesc)
    ccall((:cudnnDestroyPoolingDescriptor, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t,), poolingDesc)
end

function cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y)
    ccall((:cudnnPoolingForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnPoolingDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    ccall((:cudnnPoolingBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnPoolingDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
end

function cudnnCreateActivationDescriptor(activationDesc)
    ccall((:cudnnCreateActivationDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnActivationDescriptor_t},), activationDesc)
end

function cudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef)
    ccall((:cudnnSetActivationDescriptor, libcudnn), cudnnStatus_t, (cudnnActivationDescriptor_t, cudnnActivationMode_t, cudnnNanPropagation_t, Cdouble), activationDesc, mode, reluNanOpt, coef)
end

function cudnnGetActivationDescriptor(activationDesc, mode, reluNanOpt, coef)
    ccall((:cudnnGetActivationDescriptor, libcudnn), cudnnStatus_t, (cudnnActivationDescriptor_t, Ptr{cudnnActivationMode_t}, Ptr{cudnnNanPropagation_t}, Ptr{Cdouble}), activationDesc, mode, reluNanOpt, coef)
end

function cudnnDestroyActivationDescriptor(activationDesc)
    ccall((:cudnnDestroyActivationDescriptor, libcudnn), cudnnStatus_t, (cudnnActivationDescriptor_t,), activationDesc)
end

function cudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y)
    ccall((:cudnnActivationForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnActivationDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, activationDesc, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    ccall((:cudnnActivationBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnActivationDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
end

function cudnnCreateLRNDescriptor(normDesc)
    ccall((:cudnnCreateLRNDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnLRNDescriptor_t},), normDesc)
end

function cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
    ccall((:cudnnSetLRNDescriptor, libcudnn), cudnnStatus_t, (cudnnLRNDescriptor_t, UInt32, Cdouble, Cdouble, Cdouble), normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
end

function cudnnGetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
    ccall((:cudnnGetLRNDescriptor, libcudnn), cudnnStatus_t, (cudnnLRNDescriptor_t, Ptr{UInt32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
end

function cudnnDestroyLRNDescriptor(lrnDesc)
    ccall((:cudnnDestroyLRNDescriptor, libcudnn), cudnnStatus_t, (cudnnLRNDescriptor_t,), lrnDesc)
end

function cudnnLRNCrossChannelForward(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y)
    ccall((:cudnnLRNCrossChannelForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnLRNCrossChannelBackward(handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    ccall((:cudnnLRNCrossChannelBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
end

function cudnnDivisiveNormalizationForward(handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y)
    ccall((:cudnnDivisiveNormalizationForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y)
end

function cudnnDivisiveNormalizationBackward(handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dXdMeansDesc, dx, dMeans)
    ccall((:cudnnDivisiveNormalizationBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}), handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dXdMeansDesc, dx, dMeans)
end

function cudnnDeriveBNTensorDescriptor(derivedBnDesc, xDesc, mode)
    ccall((:cudnnDeriveBNTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnBatchNormMode_t), derivedBnDesc, xDesc, mode)
end

function cudnnBatchNormalizationForwardTraining(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance)
    ccall((:cudnnBatchNormalizationForwardTraining, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Cdouble, Ptr{Void}, Ptr{Void}, Cdouble, Ptr{Void}, Ptr{Void}), handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance)
end

function cudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon)
    ccall((:cudnnBatchNormalizationForwardInference, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, Cdouble), handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon)
end

function cudnnBatchNormalizationBackward(handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance)
    ccall((:cudnnBatchNormalizationBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, Cdouble, Ptr{Void}, Ptr{Void}), handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance)
end

function cudnnCreateSpatialTransformerDescriptor(stDesc)
    ccall((:cudnnCreateSpatialTransformerDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnSpatialTransformerDescriptor_t},), stDesc)
end

function cudnnSetSpatialTransformerNdDescriptor(stDesc, samplerType, dataType, nbDims, dimA)
    ccall((:cudnnSetSpatialTransformerNdDescriptor, libcudnn), cudnnStatus_t, (cudnnSpatialTransformerDescriptor_t, cudnnSamplerType_t, cudnnDataType_t, Cint, Ptr{Cint}), stDesc, samplerType, dataType, nbDims, dimA)
end

function cudnnDestroySpatialTransformerDescriptor(stDesc)
    ccall((:cudnnDestroySpatialTransformerDescriptor, libcudnn), cudnnStatus_t, (cudnnSpatialTransformerDescriptor_t,), stDesc)
end

function cudnnSpatialTfGridGeneratorForward(handle, stDesc, theta, grid)
    ccall((:cudnnSpatialTfGridGeneratorForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Void}, Ptr{Void}), handle, stDesc, theta, grid)
end

function cudnnSpatialTfGridGeneratorBackward(handle, stDesc, dgrid, dtheta)
    ccall((:cudnnSpatialTfGridGeneratorBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Void}, Ptr{Void}), handle, stDesc, dgrid, dtheta)
end

function cudnnSpatialTfSamplerForward(handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y)
    ccall((:cudnnSpatialTfSamplerForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y)
end

function cudnnSpatialTfSamplerBackward(handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid, betaDgrid, dgrid)
    ccall((:cudnnSpatialTfSamplerBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}), handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid, betaDgrid, dgrid)
end

function cudnnCreateDropoutDescriptor(dropoutDesc)
    ccall((:cudnnCreateDropoutDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnDropoutDescriptor_t},), dropoutDesc)
end

function cudnnDestroyDropoutDescriptor(dropoutDesc)
    ccall((:cudnnDestroyDropoutDescriptor, libcudnn), cudnnStatus_t, (cudnnDropoutDescriptor_t,), dropoutDesc)
end

function cudnnDropoutGetStatesSize(handle, sizeInBytes)
    ccall((:cudnnDropoutGetStatesSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Csize_t}), handle, sizeInBytes)
end

function cudnnDropoutGetReserveSpaceSize(xdesc, sizeInBytes)
    ccall((:cudnnDropoutGetReserveSpaceSize, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, Ptr{Csize_t}), xdesc, sizeInBytes)
end

function cudnnSetDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
    ccall((:cudnnSetDropoutDescriptor, libcudnn), cudnnStatus_t, (cudnnDropoutDescriptor_t, cudnnHandle_t, Cfloat, Ptr{Void}, Csize_t, Culonglong), dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
end

function cudnnRestoreDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
    ccall((:cudnnRestoreDropoutDescriptor, libcudnn), cudnnStatus_t, (cudnnDropoutDescriptor_t, cudnnHandle_t, Cfloat, Ptr{Void}, Csize_t, Culonglong), dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
end

function cudnnGetDropoutDescriptor(dropoutDesc, handle, dropout, states, seed)
    ccall((:cudnnGetDropoutDescriptor, libcudnn), cudnnStatus_t, (cudnnDropoutDescriptor_t, cudnnHandle_t, Ptr{Cfloat}, Ptr{Ptr{Void}}, Ptr{Culonglong}), dropoutDesc, handle, dropout, states, seed)
end

function cudnnDropoutForward(handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes)
    ccall((:cudnnDropoutForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnDropoutDescriptor_t, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Csize_t), handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnDropoutBackward(handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes)
    ccall((:cudnnDropoutBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnDropoutDescriptor_t, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Csize_t), handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnCreateRNNDescriptor(rnnDesc)
    ccall((:cudnnCreateRNNDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnRNNDescriptor_t},), rnnDesc)
end

function cudnnDestroyRNNDescriptor(rnnDesc)
    ccall((:cudnnDestroyRNNDescriptor, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t,), rnnDesc)
end

function cudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, plan)
    ccall((:cudnnCreatePersistentRNNPlan, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, Cint, cudnnDataType_t, Ptr{cudnnPersistentRNNPlan_t}), rnnDesc, minibatch, dataType, plan)
end

function cudnnSetPersistentRNNPlan(rnnDesc, plan)
    ccall((:cudnnSetPersistentRNNPlan, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, cudnnPersistentRNNPlan_t), rnnDesc, plan)
end

function cudnnDestroyPersistentRNNPlan(plan)
    ccall((:cudnnDestroyPersistentRNNPlan, libcudnn), cudnnStatus_t, (cudnnPersistentRNNPlan_t,), plan)
end

function cudnnSetRNNDescriptor(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, dataType)
    ccall((:cudnnSetRNNDescriptor, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Cint, cudnnDropoutDescriptor_t, cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t, cudnnRNNAlgo_t, cudnnDataType_t), handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, dataType)
end

function cudnnGetRNNDescriptor(cudnnHandle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, dataType)
    ccall((:cudnnGetRNNDescriptor, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{cudnnDropoutDescriptor_t}, Ptr{cudnnRNNInputMode_t}, Ptr{cudnnDirectionMode_t}, Ptr{cudnnRNNMode_t}, Ptr{cudnnRNNAlgo_t}, Ptr{cudnnDataType_t}), cudnnHandle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, dataType)
end

function cudnnSetRNNMatrixMathType(desc, math)
    ccall((:cudnnSetRNNMatrixMathType, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, cudnnMathType_t), desc, math)
end

function cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes)
    ccall((:cudnnGetRNNWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ptr{Csize_t}), handle, rnnDesc, seqLength, xDesc, sizeInBytes)
end

function cudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes)
    ccall((:cudnnGetRNNTrainingReserveSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ptr{Csize_t}), handle, rnnDesc, seqLength, xDesc, sizeInBytes)
end

function cudnnGetRNNParamsSize(handle, rnnDesc, xDesc, sizeInBytes, dataType)
    ccall((:cudnnGetRNNParamsSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnTensorDescriptor_t, Ptr{Csize_t}, cudnnDataType_t), handle, rnnDesc, xDesc, sizeInBytes, dataType)
end

function cudnnGetRNNLinLayerMatrixParams(handle, rnnDesc, layer, xDesc, wDesc, w, linLayerID, linLayerMatDesc, linLayerMat)
    ccall((:cudnnGetRNNLinLayerMatrixParams, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, Ptr{Void}, Cint, cudnnFilterDescriptor_t, Ptr{Ptr{Void}}), handle, rnnDesc, layer, xDesc, wDesc, w, linLayerID, linLayerMatDesc, linLayerMat)
end

function cudnnGetRNNLinLayerBiasParams(handle, rnnDesc, layer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc, linLayerBias)
    ccall((:cudnnGetRNNLinLayerBiasParams, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, Ptr{Void}, Cint, cudnnFilterDescriptor_t, Ptr{Ptr{Void}}), handle, rnnDesc, layer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc, linLayerBias)
end

function cudnnRNNForwardInference(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes)
    ccall((:cudnnRNNForwardInference, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Csize_t), handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes)
end

function cudnnRNNForwardTraining(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    ccall((:cudnnRNNForwardTraining, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Csize_t, Ptr{Void}, Csize_t), handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnRNNBackwardData(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    ccall((:cudnnRNNBackwardData, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Csize_t, Ptr{Void}, Csize_t), handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnRNNBackwardWeights(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
    ccall((:cudnnRNNBackwardWeights, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, Ptr{Void}, Csize_t, cudnnFilterDescriptor_t, Ptr{Void}, Ptr{Void}, Csize_t), handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnCreateCTCLossDescriptor(ctcLossDesc)
    ccall((:cudnnCreateCTCLossDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnCTCLossDescriptor_t},), ctcLossDesc)
end

function cudnnSetCTCLossDescriptor(ctcLossDesc, compType)
    ccall((:cudnnSetCTCLossDescriptor, libcudnn), cudnnStatus_t, (cudnnCTCLossDescriptor_t, cudnnDataType_t), ctcLossDesc, compType)
end

function cudnnGetCTCLossDescriptor(ctcLossDesc, compType)
    ccall((:cudnnGetCTCLossDescriptor, libcudnn), cudnnStatus_t, (cudnnCTCLossDescriptor_t, Ptr{cudnnDataType_t}), ctcLossDesc, compType)
end

function cudnnDestroyCTCLossDescriptor(ctcLossDesc)
    ccall((:cudnnDestroyCTCLossDescriptor, libcudnn), cudnnStatus_t, (cudnnCTCLossDescriptor_t,), ctcLossDesc)
end

function cudnnCTCLoss(handle, probsDesc, probs, labels, labelLengths, inputLengths, costs, gradientsDesc, gradients, algo, ctcLossDesc, workspace, workSpaceSizeInBytes)
    ccall((:cudnnCTCLoss, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, Ptr{Void}, Csize_t), handle, probsDesc, probs, labels, labelLengths, inputLengths, costs, gradientsDesc, gradients, algo, ctcLossDesc, workspace, workSpaceSizeInBytes)
end

function cudnnGetCTCLossWorkspaceSize(handle, probsDesc, gradientsDesc, labels, labelLengths, inputLengths, algo, ctcLossDesc, sizeInBytes)
    ccall((:cudnnGetCTCLossWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, Ptr{Csize_t}), handle, probsDesc, gradientsDesc, labels, labelLengths, inputLengths, algo, ctcLossDesc, sizeInBytes)
end

function cudnnSetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, dataType)
    ccall((:cudnnSetRNNDescriptor_v6, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Cint, cudnnDropoutDescriptor_t, cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t, cudnnRNNAlgo_t, cudnnDataType_t), handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, dataType)
end

function cudnnSetRNNDescriptor_v5(rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, dataType)
    ccall((:cudnnSetRNNDescriptor_v5, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, Cint, Cint, cudnnDropoutDescriptor_t, cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t, cudnnDataType_t), rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, dataType)
end
