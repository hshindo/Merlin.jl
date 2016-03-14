# Julia wrapper for header: /usr/local/cuda-7.5/include/driver_types.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0

# Julia wrapper for header: /usr/local/cuda-7.5/include/vector_types.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0

# Julia wrapper for header: /usr/local/cuda-7.5/include/cuda_runtime_api.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function cudaDeviceReset()
    check_error(ccall((:cudaDeviceReset,libcuda),cudaError_t,()))
end

function cudaDeviceSynchronize()
    check_error(ccall((:cudaDeviceSynchronize,libcuda),cudaError_t,()))
end

function cudaDeviceSetLimit(limit,value)
    check_error(ccall((:cudaDeviceSetLimit,libcuda),cudaError_t,(cudaLimit,Csize_t),limit,value))
end

function cudaDeviceGetLimit(pValue,limit)
    check_error(ccall((:cudaDeviceGetLimit,libcuda),cudaError_t,(Ptr{Csize_t},cudaLimit),pValue,limit))
end

function cudaDeviceGetCacheConfig(pCacheConfig)
    check_error(ccall((:cudaDeviceGetCacheConfig,libcuda),cudaError_t,(Ptr{cudaFuncCache},),pCacheConfig))
end

function cudaDeviceGetStreamPriorityRange(leastPriority,greatestPriority)
    check_error(ccall((:cudaDeviceGetStreamPriorityRange,libcuda),cudaError_t,(Ptr{Cint},Ptr{Cint}),leastPriority,greatestPriority))
end

function cudaDeviceSetCacheConfig(cacheConfig)
    check_error(ccall((:cudaDeviceSetCacheConfig,libcuda),cudaError_t,(cudaFuncCache,),cacheConfig))
end

function cudaDeviceGetSharedMemConfig(pConfig)
    check_error(ccall((:cudaDeviceGetSharedMemConfig,libcuda),cudaError_t,(Ptr{cudaSharedMemConfig},),pConfig))
end

function cudaDeviceSetSharedMemConfig(config)
    check_error(ccall((:cudaDeviceSetSharedMemConfig,libcuda),cudaError_t,(cudaSharedMemConfig,),config))
end

function cudaDeviceGetByPCIBusId(device,pciBusId)
    check_error(ccall((:cudaDeviceGetByPCIBusId,libcuda),cudaError_t,(Ptr{Cint},Ptr{UInt8}),device,pciBusId))
end

function cudaDeviceGetPCIBusId(pciBusId,len,device)
    check_error(ccall((:cudaDeviceGetPCIBusId,libcuda),cudaError_t,(Ptr{UInt8},Cint,Cint),pciBusId,len,device))
end

function cudaIpcGetEventHandle(handle,event)
    check_error(ccall((:cudaIpcGetEventHandle,libcuda),cudaError_t,(Ptr{cudaIpcEventHandle_t},cudaEvent_t),handle,event))
end

function cudaIpcOpenEventHandle(event,handle)
    check_error(ccall((:cudaIpcOpenEventHandle,libcuda),cudaError_t,(Ptr{cudaEvent_t},cudaIpcEventHandle_t),event,handle))
end

function cudaIpcGetMemHandle(handle,devPtr)
    check_error(ccall((:cudaIpcGetMemHandle,libcuda),cudaError_t,(Ptr{cudaIpcMemHandle_t},Ptr{Void}),handle,devPtr))
end

function cudaIpcOpenMemHandle(devPtr,handle,flags)
    check_error(ccall((:cudaIpcOpenMemHandle,libcuda),cudaError_t,(Ptr{Ptr{Void}},cudaIpcMemHandle_t,UInt32),devPtr,handle,flags))
end

function cudaIpcCloseMemHandle(devPtr)
    check_error(ccall((:cudaIpcCloseMemHandle,libcuda),cudaError_t,(Ptr{Void},),devPtr))
end

function cudaThreadExit()
    check_error(ccall((:cudaThreadExit,libcuda),cudaError_t,()))
end

function cudaThreadSynchronize()
    check_error(ccall((:cudaThreadSynchronize,libcuda),cudaError_t,()))
end

function cudaThreadSetLimit(limit,value)
    check_error(ccall((:cudaThreadSetLimit,libcuda),cudaError_t,(cudaLimit,Csize_t),limit,value))
end

function cudaThreadGetLimit(pValue,limit)
    check_error(ccall((:cudaThreadGetLimit,libcuda),cudaError_t,(Ptr{Csize_t},cudaLimit),pValue,limit))
end

function cudaThreadGetCacheConfig(pCacheConfig)
    check_error(ccall((:cudaThreadGetCacheConfig,libcuda),cudaError_t,(Ptr{cudaFuncCache},),pCacheConfig))
end

function cudaThreadSetCacheConfig(cacheConfig)
    check_error(ccall((:cudaThreadSetCacheConfig,libcuda),cudaError_t,(cudaFuncCache,),cacheConfig))
end

function cudaGetLastError()
    ccall((:cudaGetLastError,libcuda),cudaError_t,())
end

function cudaPeekAtLastError()
    ccall((:cudaPeekAtLastError,libcuda),cudaError_t,())
end

function cudaGetErrorName(error)
    ccall((:cudaGetErrorName,libcuda),Ptr{UInt8},(cudaError_t,),error)
end

function cudaGetErrorString(error)
    ccall((:cudaGetErrorString,libcuda),Ptr{UInt8},(cudaError_t,),error)
end

function cudaGetDeviceCount(count)
    check_error(ccall((:cudaGetDeviceCount,libcuda),cudaError_t,(Ptr{Cint},),count))
end

function cudaGetDeviceProperties(prop,device)
    check_error(ccall((:cudaGetDeviceProperties,libcuda),cudaError_t,(Ptr{cudaDeviceProp},Cint),prop,device))
end

function cudaDeviceGetAttribute(value,attr,device)
    check_error(ccall((:cudaDeviceGetAttribute,libcuda),cudaError_t,(Ptr{Cint},cudaDeviceAttr,Cint),value,attr,device))
end

function cudaChooseDevice(device,prop)
    check_error(ccall((:cudaChooseDevice,libcuda),cudaError_t,(Ptr{Cint},Ptr{cudaDeviceProp}),device,prop))
end

function cudaSetDevice(device)
    check_error(ccall((:cudaSetDevice,libcuda),cudaError_t,(Cint,),device))
end

function cudaGetDevice(device)
    check_error(ccall((:cudaGetDevice,libcuda),cudaError_t,(Ptr{Cint},),device))
end

function cudaSetValidDevices(device_arr,len)
    check_error(ccall((:cudaSetValidDevices,libcuda),cudaError_t,(Ptr{Cint},Cint),device_arr,len))
end

function cudaSetDeviceFlags(flags)
    check_error(ccall((:cudaSetDeviceFlags,libcuda),cudaError_t,(UInt32,),flags))
end

function cudaGetDeviceFlags(flags)
    check_error(ccall((:cudaGetDeviceFlags,libcuda),cudaError_t,(Ptr{UInt32},),flags))
end

function cudaStreamCreate(pStream)
    check_error(ccall((:cudaStreamCreate,libcuda),cudaError_t,(Ptr{cudaStream_t},),pStream))
end

function cudaStreamCreateWithFlags(pStream,flags)
    check_error(ccall((:cudaStreamCreateWithFlags,libcuda),cudaError_t,(Ptr{cudaStream_t},UInt32),pStream,flags))
end

function cudaStreamCreateWithPriority(pStream,flags,priority)
    check_error(ccall((:cudaStreamCreateWithPriority,libcuda),cudaError_t,(Ptr{cudaStream_t},UInt32,Cint),pStream,flags,priority))
end

function cudaStreamGetPriority(hStream,priority)
    check_error(ccall((:cudaStreamGetPriority,libcuda),cudaError_t,(cudaStream_t,Ptr{Cint}),hStream,priority))
end

function cudaStreamGetFlags(hStream,flags)
    check_error(ccall((:cudaStreamGetFlags,libcuda),cudaError_t,(cudaStream_t,Ptr{UInt32}),hStream,flags))
end

function cudaStreamDestroy(stream)
    check_error(ccall((:cudaStreamDestroy,libcuda),cudaError_t,(cudaStream_t,),stream))
end

function cudaStreamWaitEvent(stream,event,flags)
    check_error(ccall((:cudaStreamWaitEvent,libcuda),cudaError_t,(cudaStream_t,cudaEvent_t,UInt32),stream,event,flags))
end

function cudaStreamAddCallback(stream,callback,userData,flags)
    check_error(ccall((:cudaStreamAddCallback,libcuda),cudaError_t,(cudaStream_t,cudaStreamCallback_t,Ptr{Void},UInt32),stream,callback,userData,flags))
end

function cudaStreamSynchronize(stream)
    check_error(ccall((:cudaStreamSynchronize,libcuda),cudaError_t,(cudaStream_t,),stream))
end

function cudaStreamQuery(stream)
    ccall((:cudaStreamQuery,libcuda),cudaError_t,(cudaStream_t,),stream)
end

function cudaStreamAttachMemAsync(stream,devPtr,length,flags)
    check_error(ccall((:cudaStreamAttachMemAsync,libcuda),cudaError_t,(cudaStream_t,Ptr{Void},Csize_t,UInt32),stream,devPtr,length,flags))
end

function cudaEventCreate(event)
    check_error(ccall((:cudaEventCreate,libcuda),cudaError_t,(Ptr{cudaEvent_t},),event))
end

function cudaEventCreateWithFlags(event,flags)
    check_error(ccall((:cudaEventCreateWithFlags,libcuda),cudaError_t,(Ptr{cudaEvent_t},UInt32),event,flags))
end

function cudaEventRecord(event,stream)
    check_error(ccall((:cudaEventRecord,libcuda),cudaError_t,(cudaEvent_t,cudaStream_t),event,stream))
end

function cudaEventQuery(event)
    check_error(ccall((:cudaEventQuery,libcuda),cudaError_t,(cudaEvent_t,),event))
end

function cudaEventSynchronize(event)
    check_error(ccall((:cudaEventSynchronize,libcuda),cudaError_t,(cudaEvent_t,),event))
end

function cudaEventDestroy(event)
    check_error(ccall((:cudaEventDestroy,libcuda),cudaError_t,(cudaEvent_t,),event))
end

function cudaEventElapsedTime(ms,start,_end)
    check_error(ccall((:cudaEventElapsedTime,libcuda),cudaError_t,(Ptr{Cfloat},cudaEvent_t,cudaEvent_t),ms,start,_end))
end

function cudaLaunchKernel(func,gridDim,blockDim,args,sharedMem,stream)
    check_error(ccall((:cudaLaunchKernel,libcuda),cudaError_t,(Ptr{Void},dim3,dim3,Ptr{Ptr{Void}},Csize_t,cudaStream_t),func,gridDim,blockDim,args,sharedMem,stream))
end

function cudaFuncSetCacheConfig(func,cacheConfig)
    check_error(ccall((:cudaFuncSetCacheConfig,libcuda),cudaError_t,(Ptr{Void},cudaFuncCache),func,cacheConfig))
end

function cudaFuncSetSharedMemConfig(func,config)
    check_error(ccall((:cudaFuncSetSharedMemConfig,libcuda),cudaError_t,(Ptr{Void},cudaSharedMemConfig),func,config))
end

function cudaFuncGetAttributes(attr,func)
    check_error(ccall((:cudaFuncGetAttributes,libcuda),cudaError_t,(Ptr{cudaFuncAttributes},Ptr{Void}),attr,func))
end

function cudaSetDoubleForDevice(d)
    check_error(ccall((:cudaSetDoubleForDevice,libcuda),cudaError_t,(Ptr{Cdouble},),d))
end

function cudaSetDoubleForHost(d)
    check_error(ccall((:cudaSetDoubleForHost,libcuda),cudaError_t,(Ptr{Cdouble},),d))
end

function cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks,func,blockSize,dynamicSMemSize)
    check_error(ccall((:cudaOccupancyMaxActiveBlocksPerMultiprocessor,libcuda),cudaError_t,(Ptr{Cint},Ptr{Void},Cint,Csize_t),numBlocks,func,blockSize,dynamicSMemSize))
end

function cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks,func,blockSize,dynamicSMemSize,flags)
    check_error(ccall((:cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,libcuda),cudaError_t,(Ptr{Cint},Ptr{Void},Cint,Csize_t,UInt32),numBlocks,func,blockSize,dynamicSMemSize,flags))
end

function cudaConfigureCall(gridDim,blockDim,sharedMem,stream)
    check_error(ccall((:cudaConfigureCall,libcuda),cudaError_t,(dim3,dim3,Csize_t,cudaStream_t),gridDim,blockDim,sharedMem,stream))
end

function cudaSetupArgument(arg,size,offset)
    check_error(ccall((:cudaSetupArgument,libcuda),cudaError_t,(Ptr{Void},Csize_t,Csize_t),arg,size,offset))
end

function cudaLaunch(func)
    check_error(ccall((:cudaLaunch,libcuda),cudaError_t,(Ptr{Void},),func))
end

function cudaMallocManaged(devPtr,size,flags)
    check_error(ccall((:cudaMallocManaged,libcuda),cudaError_t,(Ptr{Ptr{Void}},Csize_t,UInt32),devPtr,size,flags))
end

function cudaMalloc(devPtr,size)
    check_error(ccall((:cudaMalloc,libcuda),cudaError_t,(Ptr{Ptr{Void}},Csize_t),devPtr,size))
end

function cudaMallocHost(ptr,size)
    check_error(ccall((:cudaMallocHost,libcuda),cudaError_t,(Ptr{Ptr{Void}},Csize_t),ptr,size))
end

function cudaMallocPitch(devPtr,pitch,width,height)
    check_error(ccall((:cudaMallocPitch,libcuda),cudaError_t,(Ptr{Ptr{Void}},Ptr{Csize_t},Csize_t,Csize_t),devPtr,pitch,width,height))
end

function cudaMallocArray(array,desc,width,height,flags)
    check_error(ccall((:cudaMallocArray,libcuda),cudaError_t,(Ptr{cudaArray_t},Ptr{cudaChannelFormatDesc},Csize_t,Csize_t,UInt32),array,desc,width,height,flags))
end

function cudaFree(devPtr)
    check_error(ccall((:cudaFree,libcuda),cudaError_t,(Ptr{Void},),devPtr))
end

function cudaFreeHost(ptr)
    check_error(ccall((:cudaFreeHost,libcuda),cudaError_t,(Ptr{Void},),ptr))
end

function cudaFreeArray(array)
    check_error(ccall((:cudaFreeArray,libcuda),cudaError_t,(cudaArray_t,),array))
end

function cudaFreeMipmappedArray(mipmappedArray)
    check_error(ccall((:cudaFreeMipmappedArray,libcuda),cudaError_t,(cudaMipmappedArray_t,),mipmappedArray))
end

function cudaHostAlloc(pHost,size,flags)
    check_error(ccall((:cudaHostAlloc,libcuda),cudaError_t,(Ptr{Ptr{Void}},Csize_t,UInt32),pHost,size,flags))
end

function cudaHostRegister(ptr,size,flags)
    check_error(ccall((:cudaHostRegister,libcuda),cudaError_t,(Ptr{Void},Csize_t,UInt32),ptr,size,flags))
end

function cudaHostUnregister(ptr)
    check_error(ccall((:cudaHostUnregister,libcuda),cudaError_t,(Ptr{Void},),ptr))
end

function cudaHostGetDevicePointer(pDevice,pHost,flags)
    check_error(ccall((:cudaHostGetDevicePointer,libcuda),cudaError_t,(Ptr{Ptr{Void}},Ptr{Void},UInt32),pDevice,pHost,flags))
end

function cudaHostGetFlags(pFlags,pHost)
    check_error(ccall((:cudaHostGetFlags,libcuda),cudaError_t,(Ptr{UInt32},Ptr{Void}),pFlags,pHost))
end

function cudaMalloc3D(pitchedDevPtr,extent)
    check_error(ccall((:cudaMalloc3D,libcuda),cudaError_t,(Ptr{cudaPitchedPtr},cudaExtent),pitchedDevPtr,extent))
end

function cudaMalloc3DArray(array,desc,extent,flags)
    check_error(ccall((:cudaMalloc3DArray,libcuda),cudaError_t,(Ptr{cudaArray_t},Ptr{cudaChannelFormatDesc},cudaExtent,UInt32),array,desc,extent,flags))
end

function cudaMallocMipmappedArray(mipmappedArray,desc,extent,numLevels,flags)
    check_error(ccall((:cudaMallocMipmappedArray,libcuda),cudaError_t,(Ptr{cudaMipmappedArray_t},Ptr{cudaChannelFormatDesc},cudaExtent,UInt32,UInt32),mipmappedArray,desc,extent,numLevels,flags))
end

function cudaGetMipmappedArrayLevel(levelArray,mipmappedArray,level)
    check_error(ccall((:cudaGetMipmappedArrayLevel,libcuda),cudaError_t,(Ptr{cudaArray_t},cudaMipmappedArray_const_t,UInt32),levelArray,mipmappedArray,level))
end

function cudaMemcpy3D(p)
    check_error(ccall((:cudaMemcpy3D,libcuda),cudaError_t,(Ptr{cudaMemcpy3DParms},),p))
end

function cudaMemcpy3DPeer(p)
    check_error(ccall((:cudaMemcpy3DPeer,libcuda),cudaError_t,(Ptr{cudaMemcpy3DPeerParms},),p))
end

function cudaMemcpy3DAsync(p,stream)
    check_error(ccall((:cudaMemcpy3DAsync,libcuda),cudaError_t,(Ptr{cudaMemcpy3DParms},cudaStream_t),p,stream))
end

function cudaMemcpy3DPeerAsync(p,stream)
    check_error(ccall((:cudaMemcpy3DPeerAsync,libcuda),cudaError_t,(Ptr{cudaMemcpy3DPeerParms},cudaStream_t),p,stream))
end

function cudaMemGetInfo(free,total)
    check_error(ccall((:cudaMemGetInfo,libcuda),cudaError_t,(Ptr{Csize_t},Ptr{Csize_t}),free,total))
end

function cudaArrayGetInfo(desc,extent,flags,array)
    check_error(ccall((:cudaArrayGetInfo,libcuda),cudaError_t,(Ptr{cudaChannelFormatDesc},Ptr{cudaExtent},Ptr{UInt32},cudaArray_t),desc,extent,flags,array))
end

function cudaMemcpy(dst,src,count,kind)
    check_error(ccall((:cudaMemcpy,libcuda),cudaError_t,(Ptr{Void},Ptr{Void},Csize_t,cudaMemcpyKind),dst,src,count,kind))
end

function cudaMemcpyPeer(dst,dstDevice,src,srcDevice,count)
    check_error(ccall((:cudaMemcpyPeer,libcuda),cudaError_t,(Ptr{Void},Cint,Ptr{Void},Cint,Csize_t),dst,dstDevice,src,srcDevice,count))
end

function cudaMemcpyToArray(dst,wOffset,hOffset,src,count,kind)
    check_error(ccall((:cudaMemcpyToArray,libcuda),cudaError_t,(cudaArray_t,Csize_t,Csize_t,Ptr{Void},Csize_t,cudaMemcpyKind),dst,wOffset,hOffset,src,count,kind))
end

function cudaMemcpyFromArray(dst,src,wOffset,hOffset,count,kind)
    check_error(ccall((:cudaMemcpyFromArray,libcuda),cudaError_t,(Ptr{Void},cudaArray_const_t,Csize_t,Csize_t,Csize_t,cudaMemcpyKind),dst,src,wOffset,hOffset,count,kind))
end

function cudaMemcpyArrayToArray(dst,wOffsetDst,hOffsetDst,src,wOffsetSrc,hOffsetSrc,count,kind)
    check_error(ccall((:cudaMemcpyArrayToArray,libcuda),cudaError_t,(cudaArray_t,Csize_t,Csize_t,cudaArray_const_t,Csize_t,Csize_t,Csize_t,cudaMemcpyKind),dst,wOffsetDst,hOffsetDst,src,wOffsetSrc,hOffsetSrc,count,kind))
end

function cudaMemcpy2D(dst,dpitch,src,spitch,width,height,kind)
    check_error(ccall((:cudaMemcpy2D,libcuda),cudaError_t,(Ptr{Void},Csize_t,Ptr{Void},Csize_t,Csize_t,Csize_t,cudaMemcpyKind),dst,dpitch,src,spitch,width,height,kind))
end

function cudaMemcpy2DToArray(dst,wOffset,hOffset,src,spitch,width,height,kind)
    check_error(ccall((:cudaMemcpy2DToArray,libcuda),cudaError_t,(cudaArray_t,Csize_t,Csize_t,Ptr{Void},Csize_t,Csize_t,Csize_t,cudaMemcpyKind),dst,wOffset,hOffset,src,spitch,width,height,kind))
end

function cudaMemcpy2DFromArray(dst,dpitch,src,wOffset,hOffset,width,height,kind)
    check_error(ccall((:cudaMemcpy2DFromArray,libcuda),cudaError_t,(Ptr{Void},Csize_t,cudaArray_const_t,Csize_t,Csize_t,Csize_t,Csize_t,cudaMemcpyKind),dst,dpitch,src,wOffset,hOffset,width,height,kind))
end

function cudaMemcpy2DArrayToArray(dst,wOffsetDst,hOffsetDst,src,wOffsetSrc,hOffsetSrc,width,height,kind)
    check_error(ccall((:cudaMemcpy2DArrayToArray,libcuda),cudaError_t,(cudaArray_t,Csize_t,Csize_t,cudaArray_const_t,Csize_t,Csize_t,Csize_t,Csize_t,cudaMemcpyKind),dst,wOffsetDst,hOffsetDst,src,wOffsetSrc,hOffsetSrc,width,height,kind))
end

function cudaMemcpyToSymbol(symbol,src,count,offset,kind)
    check_error(ccall((:cudaMemcpyToSymbol,libcuda),cudaError_t,(Ptr{Void},Ptr{Void},Csize_t,Csize_t,cudaMemcpyKind),symbol,src,count,offset,kind))
end

function cudaMemcpyFromSymbol(dst,symbol,count,offset,kind)
    check_error(ccall((:cudaMemcpyFromSymbol,libcuda),cudaError_t,(Ptr{Void},Ptr{Void},Csize_t,Csize_t,cudaMemcpyKind),dst,symbol,count,offset,kind))
end

function cudaMemcpyAsync(dst,src,count,kind,stream)
    check_error(ccall((:cudaMemcpyAsync,libcuda),cudaError_t,(Ptr{Void},Ptr{Void},Csize_t,cudaMemcpyKind,cudaStream_t),dst,src,count,kind,stream))
end

function cudaMemcpyPeerAsync(dst,dstDevice,src,srcDevice,count,stream)
    check_error(ccall((:cudaMemcpyPeerAsync,libcuda),cudaError_t,(Ptr{Void},Cint,Ptr{Void},Cint,Csize_t,cudaStream_t),dst,dstDevice,src,srcDevice,count,stream))
end

function cudaMemcpyToArrayAsync(dst,wOffset,hOffset,src,count,kind,stream)
    check_error(ccall((:cudaMemcpyToArrayAsync,libcuda),cudaError_t,(cudaArray_t,Csize_t,Csize_t,Ptr{Void},Csize_t,cudaMemcpyKind,cudaStream_t),dst,wOffset,hOffset,src,count,kind,stream))
end

function cudaMemcpyFromArrayAsync(dst,src,wOffset,hOffset,count,kind,stream)
    check_error(ccall((:cudaMemcpyFromArrayAsync,libcuda),cudaError_t,(Ptr{Void},cudaArray_const_t,Csize_t,Csize_t,Csize_t,cudaMemcpyKind,cudaStream_t),dst,src,wOffset,hOffset,count,kind,stream))
end

function cudaMemcpy2DAsync(dst,dpitch,src,spitch,width,height,kind,stream)
    check_error(ccall((:cudaMemcpy2DAsync,libcuda),cudaError_t,(Ptr{Void},Csize_t,Ptr{Void},Csize_t,Csize_t,Csize_t,cudaMemcpyKind,cudaStream_t),dst,dpitch,src,spitch,width,height,kind,stream))
end

function cudaMemcpy2DToArrayAsync(dst,wOffset,hOffset,src,spitch,width,height,kind,stream)
    check_error(ccall((:cudaMemcpy2DToArrayAsync,libcuda),cudaError_t,(cudaArray_t,Csize_t,Csize_t,Ptr{Void},Csize_t,Csize_t,Csize_t,cudaMemcpyKind,cudaStream_t),dst,wOffset,hOffset,src,spitch,width,height,kind,stream))
end

function cudaMemcpy2DFromArrayAsync(dst,dpitch,src,wOffset,hOffset,width,height,kind,stream)
    check_error(ccall((:cudaMemcpy2DFromArrayAsync,libcuda),cudaError_t,(Ptr{Void},Csize_t,cudaArray_const_t,Csize_t,Csize_t,Csize_t,Csize_t,cudaMemcpyKind,cudaStream_t),dst,dpitch,src,wOffset,hOffset,width,height,kind,stream))
end

function cudaMemcpyToSymbolAsync(symbol,src,count,offset,kind,stream)
    check_error(ccall((:cudaMemcpyToSymbolAsync,libcuda),cudaError_t,(Ptr{Void},Ptr{Void},Csize_t,Csize_t,cudaMemcpyKind,cudaStream_t),symbol,src,count,offset,kind,stream))
end

function cudaMemcpyFromSymbolAsync(dst,symbol,count,offset,kind,stream)
    check_error(ccall((:cudaMemcpyFromSymbolAsync,libcuda),cudaError_t,(Ptr{Void},Ptr{Void},Csize_t,Csize_t,cudaMemcpyKind,cudaStream_t),dst,symbol,count,offset,kind,stream))
end

function cudaMemset(devPtr,value,count)
    check_error(ccall((:cudaMemset,libcuda),cudaError_t,(Ptr{Void},Cint,Csize_t),devPtr,value,count))
end

function cudaMemset2D(devPtr,pitch,value,width,height)
    check_error(ccall((:cudaMemset2D,libcuda),cudaError_t,(Ptr{Void},Csize_t,Cint,Csize_t,Csize_t),devPtr,pitch,value,width,height))
end

function cudaMemset3D(pitchedDevPtr,value,extent)
    check_error(ccall((:cudaMemset3D,libcuda),cudaError_t,(cudaPitchedPtr,Cint,cudaExtent),pitchedDevPtr,value,extent))
end

function cudaMemsetAsync(devPtr,value,count,stream)
    check_error(ccall((:cudaMemsetAsync,libcuda),cudaError_t,(Ptr{Void},Cint,Csize_t,cudaStream_t),devPtr,value,count,stream))
end

function cudaMemset2DAsync(devPtr,pitch,value,width,height,stream)
    check_error(ccall((:cudaMemset2DAsync,libcuda),cudaError_t,(Ptr{Void},Csize_t,Cint,Csize_t,Csize_t,cudaStream_t),devPtr,pitch,value,width,height,stream))
end

function cudaMemset3DAsync(pitchedDevPtr,value,extent,stream)
    check_error(ccall((:cudaMemset3DAsync,libcuda),cudaError_t,(cudaPitchedPtr,Cint,cudaExtent,cudaStream_t),pitchedDevPtr,value,extent,stream))
end

function cudaGetSymbolAddress(devPtr,symbol)
    check_error(ccall((:cudaGetSymbolAddress,libcuda),cudaError_t,(Ptr{Ptr{Void}},Ptr{Void}),devPtr,symbol))
end

function cudaGetSymbolSize(size,symbol)
    check_error(ccall((:cudaGetSymbolSize,libcuda),cudaError_t,(Ptr{Csize_t},Ptr{Void}),size,symbol))
end

function cudaPointerGetAttributes(attributes,ptr)
    check_error(ccall((:cudaPointerGetAttributes,libcuda),cudaError_t,(Ptr{cudaPointerAttributes},Ptr{Void}),attributes,ptr))
end

function cudaDeviceCanAccessPeer(canAccessPeer,device,peerDevice)
    check_error(ccall((:cudaDeviceCanAccessPeer,libcuda),cudaError_t,(Ptr{Cint},Cint,Cint),canAccessPeer,device,peerDevice))
end

function cudaDeviceEnablePeerAccess(peerDevice,flags)
    check_error(ccall((:cudaDeviceEnablePeerAccess,libcuda),cudaError_t,(Cint,UInt32),peerDevice,flags))
end

function cudaDeviceDisablePeerAccess(peerDevice)
    check_error(ccall((:cudaDeviceDisablePeerAccess,libcuda),cudaError_t,(Cint,),peerDevice))
end

function cudaGraphicsUnregisterResource(resource)
    check_error(ccall((:cudaGraphicsUnregisterResource,libcuda),cudaError_t,(cudaGraphicsResource_t,),resource))
end

function cudaGraphicsResourceSetMapFlags(resource,flags)
    check_error(ccall((:cudaGraphicsResourceSetMapFlags,libcuda),cudaError_t,(cudaGraphicsResource_t,UInt32),resource,flags))
end

function cudaGraphicsMapResources(count,resources,stream)
    check_error(ccall((:cudaGraphicsMapResources,libcuda),cudaError_t,(Cint,Ptr{cudaGraphicsResource_t},cudaStream_t),count,resources,stream))
end

function cudaGraphicsUnmapResources(count,resources,stream)
    check_error(ccall((:cudaGraphicsUnmapResources,libcuda),cudaError_t,(Cint,Ptr{cudaGraphicsResource_t},cudaStream_t),count,resources,stream))
end

function cudaGraphicsResourceGetMappedPointer(devPtr,size,resource)
    check_error(ccall((:cudaGraphicsResourceGetMappedPointer,libcuda),cudaError_t,(Ptr{Ptr{Void}},Ptr{Csize_t},cudaGraphicsResource_t),devPtr,size,resource))
end

function cudaGraphicsSubResourceGetMappedArray(array,resource,arrayIndex,mipLevel)
    check_error(ccall((:cudaGraphicsSubResourceGetMappedArray,libcuda),cudaError_t,(Ptr{cudaArray_t},cudaGraphicsResource_t,UInt32,UInt32),array,resource,arrayIndex,mipLevel))
end

function cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray,resource)
    check_error(ccall((:cudaGraphicsResourceGetMappedMipmappedArray,libcuda),cudaError_t,(Ptr{cudaMipmappedArray_t},cudaGraphicsResource_t),mipmappedArray,resource))
end

function cudaGetChannelDesc(desc,array)
    check_error(ccall((:cudaGetChannelDesc,libcuda),cudaError_t,(Ptr{cudaChannelFormatDesc},cudaArray_const_t),desc,array))
end

function cudaCreateChannelDesc(x,y,z,w,f)
    ccall((:cudaCreateChannelDesc,libcuda),cudaChannelFormatDesc,(Cint,Cint,Cint,Cint,cudaChannelFormatKind),x,y,z,w,f)
end

function cudaBindTexture(offset,texref,devPtr,desc,size)
    check_error(ccall((:cudaBindTexture,libcuda),cudaError_t,(Ptr{Csize_t},Ptr{textureReference},Ptr{Void},Ptr{cudaChannelFormatDesc},Csize_t),offset,texref,devPtr,desc,size))
end

function cudaBindTexture2D(offset,texref,devPtr,desc,width,height,pitch)
    check_error(ccall((:cudaBindTexture2D,libcuda),cudaError_t,(Ptr{Csize_t},Ptr{textureReference},Ptr{Void},Ptr{cudaChannelFormatDesc},Csize_t,Csize_t,Csize_t),offset,texref,devPtr,desc,width,height,pitch))
end

function cudaBindTextureToArray(texref,array,desc)
    check_error(ccall((:cudaBindTextureToArray,libcuda),cudaError_t,(Ptr{textureReference},cudaArray_const_t,Ptr{cudaChannelFormatDesc}),texref,array,desc))
end

function cudaBindTextureToMipmappedArray(texref,mipmappedArray,desc)
    check_error(ccall((:cudaBindTextureToMipmappedArray,libcuda),cudaError_t,(Ptr{textureReference},cudaMipmappedArray_const_t,Ptr{cudaChannelFormatDesc}),texref,mipmappedArray,desc))
end

function cudaUnbindTexture(texref)
    check_error(ccall((:cudaUnbindTexture,libcuda),cudaError_t,(Ptr{textureReference},),texref))
end

function cudaGetTextureAlignmentOffset(offset,texref)
    check_error(ccall((:cudaGetTextureAlignmentOffset,libcuda),cudaError_t,(Ptr{Csize_t},Ptr{textureReference}),offset,texref))
end

function cudaGetTextureReference(texref,symbol)
    check_error(ccall((:cudaGetTextureReference,libcuda),cudaError_t,(Ptr{Ptr{textureReference}},Ptr{Void}),texref,symbol))
end

function cudaBindSurfaceToArray(surfref,array,desc)
    check_error(ccall((:cudaBindSurfaceToArray,libcuda),cudaError_t,(Ptr{surfaceReference},cudaArray_const_t,Ptr{cudaChannelFormatDesc}),surfref,array,desc))
end

function cudaGetSurfaceReference(surfref,symbol)
    check_error(ccall((:cudaGetSurfaceReference,libcuda),cudaError_t,(Ptr{Ptr{surfaceReference}},Ptr{Void}),surfref,symbol))
end

function cudaCreateTextureObject(pTexObject,pResDesc,pTexDesc,pResViewDesc)
    check_error(ccall((:cudaCreateTextureObject,libcuda),cudaError_t,(Ptr{cudaTextureObject_t},Ptr{cudaResourceDesc},Ptr{cudaTextureDesc},Ptr{cudaResourceViewDesc}),pTexObject,pResDesc,pTexDesc,pResViewDesc))
end

function cudaDestroyTextureObject(texObject)
    check_error(ccall((:cudaDestroyTextureObject,libcuda),cudaError_t,(cudaTextureObject_t,),texObject))
end

function cudaGetTextureObjectResourceDesc(pResDesc,texObject)
    check_error(ccall((:cudaGetTextureObjectResourceDesc,libcuda),cudaError_t,(Ptr{cudaResourceDesc},cudaTextureObject_t),pResDesc,texObject))
end

function cudaGetTextureObjectTextureDesc(pTexDesc,texObject)
    check_error(ccall((:cudaGetTextureObjectTextureDesc,libcuda),cudaError_t,(Ptr{cudaTextureDesc},cudaTextureObject_t),pTexDesc,texObject))
end

function cudaGetTextureObjectResourceViewDesc(pResViewDesc,texObject)
    check_error(ccall((:cudaGetTextureObjectResourceViewDesc,libcuda),cudaError_t,(Ptr{cudaResourceViewDesc},cudaTextureObject_t),pResViewDesc,texObject))
end

function cudaCreateSurfaceObject(pSurfObject,pResDesc)
    check_error(ccall((:cudaCreateSurfaceObject,libcuda),cudaError_t,(Ptr{cudaSurfaceObject_t},Ptr{cudaResourceDesc}),pSurfObject,pResDesc))
end

function cudaDestroySurfaceObject(surfObject)
    check_error(ccall((:cudaDestroySurfaceObject,libcuda),cudaError_t,(cudaSurfaceObject_t,),surfObject))
end

function cudaGetSurfaceObjectResourceDesc(pResDesc,surfObject)
    check_error(ccall((:cudaGetSurfaceObjectResourceDesc,libcuda),cudaError_t,(Ptr{cudaResourceDesc},cudaSurfaceObject_t),pResDesc,surfObject))
end

function cudaDriverGetVersion(driverVersion)
    check_error(ccall((:cudaDriverGetVersion,libcuda),cudaError_t,(Ptr{Cint},),driverVersion))
end

function cudaRuntimeGetVersion(runtimeVersion)
    check_error(ccall((:cudaRuntimeGetVersion,libcuda),cudaError_t,(Ptr{Cint},),runtimeVersion))
end

function cudaGetExportTable(ppExportTable,pExportTableId)
    check_error(ccall((:cudaGetExportTable,libcuda),cudaError_t,(Ptr{Ptr{Void}},Ptr{cudaUUID_t}),ppExportTable,pExportTableId))
end
# Julia wrapper for header: /usr/local/cuda-7.5/include/nvrtc.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function nvrtcGetErrorString(result)
    ccall((:nvrtcGetErrorString,libcuda),Ptr{UInt8},(nvrtcResult,),result)
end

function nvrtcVersion(major,minor)
    check_nvrtc(ccall((:nvrtcVersion,libcuda),nvrtcResult,(Ptr{Cint},Ptr{Cint}),major,minor))
end

function nvrtcCreateProgram(prog,src,name,numHeaders,headers,includeNames)
    check_nvrtc(ccall((:nvrtcCreateProgram,libcuda),nvrtcResult,(Ptr{nvrtcProgram},Ptr{UInt8},Ptr{UInt8},Cint,Ptr{Ptr{UInt8}},Ptr{Ptr{UInt8}}),prog,src,name,numHeaders,headers,includeNames))
end

function nvrtcDestroyProgram(prog)
    check_nvrtc(ccall((:nvrtcDestroyProgram,libcuda),nvrtcResult,(Ptr{nvrtcProgram},),prog))
end

function nvrtcCompileProgram(prog,numOptions,options)
    check_nvrtc(ccall((:nvrtcCompileProgram,libcuda),nvrtcResult,(nvrtcProgram,Cint,Ptr{Ptr{UInt8}}),prog,numOptions,options))
end

function nvrtcGetPTXSize(prog,ptxSizeRet)
    check_nvrtc(ccall((:nvrtcGetPTXSize,libcuda),nvrtcResult,(nvrtcProgram,Ptr{Csize_t}),prog,ptxSizeRet))
end

function nvrtcGetPTX(prog,ptx)
    check_nvrtc(ccall((:nvrtcGetPTX,libcuda),nvrtcResult,(nvrtcProgram,Ptr{UInt8}),prog,ptx))
end

function nvrtcGetProgramLogSize(prog,logSizeRet)
    check_nvrtc(ccall((:nvrtcGetProgramLogSize,libcuda),nvrtcResult,(nvrtcProgram,Ptr{Csize_t}),prog,logSizeRet))
end

function nvrtcGetProgramLog(prog,log)
    check_nvrtc(ccall((:nvrtcGetProgramLog,libcuda),nvrtcResult,(nvrtcProgram,Ptr{UInt8}),prog,log))
end
