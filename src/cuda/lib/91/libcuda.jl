# Julia wrapper for header: /usr/local/cuda/include/cuda.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function cuGetErrorString(error,pStr)
    check_curesult(ccall((:cuGetErrorString,libcuda),CUresult,(CUresult,Ptr{Ptr{UInt8}}),error,pStr))
end

function cuGetErrorName(error,pStr)
    check_curesult(ccall((:cuGetErrorName,libcuda),CUresult,(CUresult,Ptr{Ptr{UInt8}}),error,pStr))
end

function cuInit(Flags)
    check_curesult(ccall((:cuInit,libcuda),CUresult,(UInt32,),Flags))
end

function cuDriverGetVersion(driverVersion)
    check_curesult(ccall((:cuDriverGetVersion,libcuda),CUresult,(Ptr{Cint},),driverVersion))
end

function cuDeviceGet(device,ordinal)
    check_curesult(ccall((:cuDeviceGet,libcuda),CUresult,(Ptr{CUdevice},Cint),device,ordinal))
end

function cuDeviceGetCount(count)
    check_curesult(ccall((:cuDeviceGetCount,libcuda),CUresult,(Ptr{Cint},),count))
end

function cuDeviceGetName(name,len,dev)
    check_curesult(ccall((:cuDeviceGetName,libcuda),CUresult,(Ptr{UInt8},Cint,CUdevice),name,len,dev))
end

function cuDeviceTotalMem_v2(bytes,dev)
    check_curesult(ccall((:cuDeviceTotalMem_v2,libcuda),CUresult,(Ptr{Csize_t},CUdevice),bytes,dev))
end

function cuDeviceGetAttribute(pi,attrib,dev)
    check_curesult(ccall((:cuDeviceGetAttribute,libcuda),CUresult,(Ptr{Cint},CUdevice_attribute,CUdevice),pi,attrib,dev))
end

function cuDeviceGetProperties(prop,dev)
    check_curesult(ccall((:cuDeviceGetProperties,libcuda),CUresult,(Ptr{CUdevprop},CUdevice),prop,dev))
end

function cuDeviceComputeCapability(major,minor,dev)
    check_curesult(ccall((:cuDeviceComputeCapability,libcuda),CUresult,(Ptr{Cint},Ptr{Cint},CUdevice),major,minor,dev))
end

function cuDevicePrimaryCtxRetain(pctx,dev)
    check_curesult(ccall((:cuDevicePrimaryCtxRetain,libcuda),CUresult,(Ptr{CUcontext},CUdevice),pctx,dev))
end

function cuDevicePrimaryCtxRelease(dev)
    check_curesult(ccall((:cuDevicePrimaryCtxRelease,libcuda),CUresult,(CUdevice,),dev))
end

function cuDevicePrimaryCtxSetFlags(dev,flags)
    check_curesult(ccall((:cuDevicePrimaryCtxSetFlags,libcuda),CUresult,(CUdevice,UInt32),dev,flags))
end

function cuDevicePrimaryCtxGetState(dev,flags,active)
    check_curesult(ccall((:cuDevicePrimaryCtxGetState,libcuda),CUresult,(CUdevice,Ptr{UInt32},Ptr{Cint}),dev,flags,active))
end

function cuDevicePrimaryCtxReset(dev)
    check_curesult(ccall((:cuDevicePrimaryCtxReset,libcuda),CUresult,(CUdevice,),dev))
end

function cuCtxCreate_v2(pctx,flags,dev)
    check_curesult(ccall((:cuCtxCreate_v2,libcuda),CUresult,(Ptr{CUcontext},UInt32,CUdevice),pctx,flags,dev))
end

function cuCtxDestroy_v2(ctx)
    check_curesult(ccall((:cuCtxDestroy_v2,libcuda),CUresult,(CUcontext,),ctx))
end

function cuCtxPushCurrent_v2(ctx)
    check_curesult(ccall((:cuCtxPushCurrent_v2,libcuda),CUresult,(CUcontext,),ctx))
end

function cuCtxPopCurrent_v2(pctx)
    check_curesult(ccall((:cuCtxPopCurrent_v2,libcuda),CUresult,(Ptr{CUcontext},),pctx))
end

function cuCtxSetCurrent(ctx)
    check_curesult(ccall((:cuCtxSetCurrent,libcuda),CUresult,(CUcontext,),ctx))
end

function cuCtxGetCurrent(pctx)
    check_curesult(ccall((:cuCtxGetCurrent,libcuda),CUresult,(Ptr{CUcontext},),pctx))
end

function cuCtxGetDevice(device)
    check_curesult(ccall((:cuCtxGetDevice,libcuda),CUresult,(Ptr{CUdevice},),device))
end

function cuCtxGetFlags(flags)
    check_curesult(ccall((:cuCtxGetFlags,libcuda),CUresult,(Ptr{UInt32},),flags))
end

function cuCtxSynchronize()
    check_curesult(ccall((:cuCtxSynchronize,libcuda),CUresult,()))
end

function cuCtxSetLimit(limit,value)
    check_curesult(ccall((:cuCtxSetLimit,libcuda),CUresult,(CUlimit,Csize_t),limit,value))
end

function cuCtxGetLimit(pvalue,limit)
    check_curesult(ccall((:cuCtxGetLimit,libcuda),CUresult,(Ptr{Csize_t},CUlimit),pvalue,limit))
end

function cuCtxGetCacheConfig(pconfig)
    check_curesult(ccall((:cuCtxGetCacheConfig,libcuda),CUresult,(Ptr{CUfunc_cache},),pconfig))
end

function cuCtxSetCacheConfig(config)
    check_curesult(ccall((:cuCtxSetCacheConfig,libcuda),CUresult,(CUfunc_cache,),config))
end

function cuCtxGetSharedMemConfig(pConfig)
    check_curesult(ccall((:cuCtxGetSharedMemConfig,libcuda),CUresult,(Ptr{CUsharedconfig},),pConfig))
end

function cuCtxSetSharedMemConfig(config)
    check_curesult(ccall((:cuCtxSetSharedMemConfig,libcuda),CUresult,(CUsharedconfig,),config))
end

function cuCtxGetApiVersion(ctx,version)
    check_curesult(ccall((:cuCtxGetApiVersion,libcuda),CUresult,(CUcontext,Ptr{UInt32}),ctx,version))
end

function cuCtxGetStreamPriorityRange(leastPriority,greatestPriority)
    check_curesult(ccall((:cuCtxGetStreamPriorityRange,libcuda),CUresult,(Ptr{Cint},Ptr{Cint}),leastPriority,greatestPriority))
end

function cuCtxAttach(pctx,flags)
    check_curesult(ccall((:cuCtxAttach,libcuda),CUresult,(Ptr{CUcontext},UInt32),pctx,flags))
end

function cuCtxDetach(ctx)
    check_curesult(ccall((:cuCtxDetach,libcuda),CUresult,(CUcontext,),ctx))
end

function cuModuleLoad(_module,fname)
    check_curesult(ccall((:cuModuleLoad,libcuda),CUresult,(Ptr{CUmodule},Ptr{UInt8}),_module,fname))
end

function cuModuleLoadData(_module,image)
    check_curesult(ccall((:cuModuleLoadData,libcuda),CUresult,(Ptr{CUmodule},Ptr{Void}),_module,image))
end

function cuModuleLoadDataEx(_module,image,numOptions,options,optionValues)
    check_curesult(ccall((:cuModuleLoadDataEx,libcuda),CUresult,(Ptr{CUmodule},Ptr{Void},UInt32,Ptr{CUjit_option},Ptr{Ptr{Void}}),_module,image,numOptions,options,optionValues))
end

function cuModuleLoadFatBinary(_module,fatCubin)
    check_curesult(ccall((:cuModuleLoadFatBinary,libcuda),CUresult,(Ptr{CUmodule},Ptr{Void}),_module,fatCubin))
end

function cuModuleUnload(hmod)
    check_curesult(ccall((:cuModuleUnload,libcuda),CUresult,(CUmodule,),hmod))
end

function cuModuleGetFunction(hfunc,hmod,name)
    check_curesult(ccall((:cuModuleGetFunction,libcuda),CUresult,(Ptr{CUfunction},CUmodule,Ptr{UInt8}),hfunc,hmod,name))
end

function cuModuleGetGlobal_v2(dptr,bytes,hmod,name)
    check_curesult(ccall((:cuModuleGetGlobal_v2,libcuda),CUresult,(Ptr{CUdeviceptr},Ptr{Csize_t},CUmodule,Ptr{UInt8}),dptr,bytes,hmod,name))
end

function cuModuleGetTexRef(pTexRef,hmod,name)
    check_curesult(ccall((:cuModuleGetTexRef,libcuda),CUresult,(Ptr{CUtexref},CUmodule,Ptr{UInt8}),pTexRef,hmod,name))
end

function cuModuleGetSurfRef(pSurfRef,hmod,name)
    check_curesult(ccall((:cuModuleGetSurfRef,libcuda),CUresult,(Ptr{CUsurfref},CUmodule,Ptr{UInt8}),pSurfRef,hmod,name))
end

function cuLinkCreate_v2(numOptions,options,optionValues,stateOut)
    check_curesult(ccall((:cuLinkCreate_v2,libcuda),CUresult,(UInt32,Ptr{CUjit_option},Ptr{Ptr{Void}},Ptr{CUlinkState}),numOptions,options,optionValues,stateOut))
end

function cuLinkAddData_v2(state,_type,data,size,name,numOptions,options,optionValues)
    check_curesult(ccall((:cuLinkAddData_v2,libcuda),CUresult,(CUlinkState,CUjitInputType,Ptr{Void},Csize_t,Ptr{UInt8},UInt32,Ptr{CUjit_option},Ptr{Ptr{Void}}),state,_type,data,size,name,numOptions,options,optionValues))
end

function cuLinkAddFile_v2(state,_type,path,numOptions,options,optionValues)
    check_curesult(ccall((:cuLinkAddFile_v2,libcuda),CUresult,(CUlinkState,CUjitInputType,Ptr{UInt8},UInt32,Ptr{CUjit_option},Ptr{Ptr{Void}}),state,_type,path,numOptions,options,optionValues))
end

function cuLinkComplete(state,cubinOut,sizeOut)
    check_curesult(ccall((:cuLinkComplete,libcuda),CUresult,(CUlinkState,Ptr{Ptr{Void}},Ptr{Csize_t}),state,cubinOut,sizeOut))
end

function cuLinkDestroy(state)
    check_curesult(ccall((:cuLinkDestroy,libcuda),CUresult,(CUlinkState,),state))
end

function cuMemGetInfo_v2(free,total)
    check_curesult(ccall((:cuMemGetInfo_v2,libcuda),CUresult,(Ptr{Csize_t},Ptr{Csize_t}),free,total))
end

function cuMemAlloc_v2(dptr,bytesize)
    check_curesult(ccall((:cuMemAlloc_v2,libcuda),CUresult,(Ptr{CUdeviceptr},Csize_t),dptr,bytesize))
end

function cuMemAllocPitch_v2(dptr,pPitch,WidthInBytes,Height,ElementSizeBytes)
    check_curesult(ccall((:cuMemAllocPitch_v2,libcuda),CUresult,(Ptr{CUdeviceptr},Ptr{Csize_t},Csize_t,Csize_t,UInt32),dptr,pPitch,WidthInBytes,Height,ElementSizeBytes))
end

function cuMemFree_v2(dptr)
    check_curesult(ccall((:cuMemFree_v2,libcuda),CUresult,(CUdeviceptr,),dptr))
end

function cuMemGetAddressRange_v2(pbase,psize,dptr)
    check_curesult(ccall((:cuMemGetAddressRange_v2,libcuda),CUresult,(Ptr{CUdeviceptr},Ptr{Csize_t},CUdeviceptr),pbase,psize,dptr))
end

function cuMemAllocHost_v2(pp,bytesize)
    check_curesult(ccall((:cuMemAllocHost_v2,libcuda),CUresult,(Ptr{Ptr{Void}},Csize_t),pp,bytesize))
end

function cuMemFreeHost(p)
    check_curesult(ccall((:cuMemFreeHost,libcuda),CUresult,(Ptr{Void},),p))
end

function cuMemHostAlloc(pp,bytesize,Flags)
    check_curesult(ccall((:cuMemHostAlloc,libcuda),CUresult,(Ptr{Ptr{Void}},Csize_t,UInt32),pp,bytesize,Flags))
end

function cuMemHostGetDevicePointer_v2(pdptr,p,Flags)
    check_curesult(ccall((:cuMemHostGetDevicePointer_v2,libcuda),CUresult,(Ptr{CUdeviceptr},Ptr{Void},UInt32),pdptr,p,Flags))
end

function cuMemHostGetFlags(pFlags,p)
    check_curesult(ccall((:cuMemHostGetFlags,libcuda),CUresult,(Ptr{UInt32},Ptr{Void}),pFlags,p))
end

function cuMemAllocManaged(dptr,bytesize,flags)
    check_curesult(ccall((:cuMemAllocManaged,libcuda),CUresult,(Ptr{CUdeviceptr},Csize_t,UInt32),dptr,bytesize,flags))
end

function cuDeviceGetByPCIBusId(dev,pciBusId)
    check_curesult(ccall((:cuDeviceGetByPCIBusId,libcuda),CUresult,(Ptr{CUdevice},Ptr{UInt8}),dev,pciBusId))
end

function cuDeviceGetPCIBusId(pciBusId,len,dev)
    check_curesult(ccall((:cuDeviceGetPCIBusId,libcuda),CUresult,(Ptr{UInt8},Cint,CUdevice),pciBusId,len,dev))
end

function cuIpcGetEventHandle(pHandle,event)
    check_curesult(ccall((:cuIpcGetEventHandle,libcuda),CUresult,(Ptr{CUipcEventHandle},CUevent),pHandle,event))
end

function cuIpcOpenEventHandle(phEvent,handle)
    check_curesult(ccall((:cuIpcOpenEventHandle,libcuda),CUresult,(Ptr{CUevent},CUipcEventHandle),phEvent,handle))
end

function cuIpcGetMemHandle(pHandle,dptr)
    check_curesult(ccall((:cuIpcGetMemHandle,libcuda),CUresult,(Ptr{CUipcMemHandle},CUdeviceptr),pHandle,dptr))
end

function cuIpcOpenMemHandle(pdptr,handle,Flags)
    check_curesult(ccall((:cuIpcOpenMemHandle,libcuda),CUresult,(Ptr{CUdeviceptr},CUipcMemHandle,UInt32),pdptr,handle,Flags))
end

function cuIpcCloseMemHandle(dptr)
    check_curesult(ccall((:cuIpcCloseMemHandle,libcuda),CUresult,(CUdeviceptr,),dptr))
end

function cuMemHostRegister_v2(p,bytesize,Flags)
    check_curesult(ccall((:cuMemHostRegister_v2,libcuda),CUresult,(Ptr{Void},Csize_t,UInt32),p,bytesize,Flags))
end

function cuMemHostUnregister(p)
    check_curesult(ccall((:cuMemHostUnregister,libcuda),CUresult,(Ptr{Void},),p))
end

function cuMemcpy(dst,src,ByteCount)
    check_curesult(ccall((:cuMemcpy,libcuda),CUresult,(CUdeviceptr,CUdeviceptr,Csize_t),dst,src,ByteCount))
end

function cuMemcpyPeer(dstDevice,dstContext,srcDevice,srcContext,ByteCount)
    check_curesult(ccall((:cuMemcpyPeer,libcuda),CUresult,(CUdeviceptr,CUcontext,CUdeviceptr,CUcontext,Csize_t),dstDevice,dstContext,srcDevice,srcContext,ByteCount))
end

function cuMemcpyHtoD_v2(dstDevice,srcHost,ByteCount)
    check_curesult(ccall((:cuMemcpyHtoD_v2,libcuda),CUresult,(CUdeviceptr,Ptr{Void},Csize_t),dstDevice,srcHost,ByteCount))
end

function cuMemcpyDtoH_v2(dstHost,srcDevice,ByteCount)
    check_curesult(ccall((:cuMemcpyDtoH_v2,libcuda),CUresult,(Ptr{Void},CUdeviceptr,Csize_t),dstHost,srcDevice,ByteCount))
end

function cuMemcpyDtoD_v2(dstDevice,srcDevice,ByteCount)
    check_curesult(ccall((:cuMemcpyDtoD_v2,libcuda),CUresult,(CUdeviceptr,CUdeviceptr,Csize_t),dstDevice,srcDevice,ByteCount))
end

function cuMemcpyDtoA_v2(dstArray,dstOffset,srcDevice,ByteCount)
    check_curesult(ccall((:cuMemcpyDtoA_v2,libcuda),CUresult,(CUarray,Csize_t,CUdeviceptr,Csize_t),dstArray,dstOffset,srcDevice,ByteCount))
end

function cuMemcpyAtoD_v2(dstDevice,srcArray,srcOffset,ByteCount)
    check_curesult(ccall((:cuMemcpyAtoD_v2,libcuda),CUresult,(CUdeviceptr,CUarray,Csize_t,Csize_t),dstDevice,srcArray,srcOffset,ByteCount))
end

function cuMemcpyHtoA_v2(dstArray,dstOffset,srcHost,ByteCount)
    check_curesult(ccall((:cuMemcpyHtoA_v2,libcuda),CUresult,(CUarray,Csize_t,Ptr{Void},Csize_t),dstArray,dstOffset,srcHost,ByteCount))
end

function cuMemcpyAtoH_v2(dstHost,srcArray,srcOffset,ByteCount)
    check_curesult(ccall((:cuMemcpyAtoH_v2,libcuda),CUresult,(Ptr{Void},CUarray,Csize_t,Csize_t),dstHost,srcArray,srcOffset,ByteCount))
end

function cuMemcpyAtoA_v2(dstArray,dstOffset,srcArray,srcOffset,ByteCount)
    check_curesult(ccall((:cuMemcpyAtoA_v2,libcuda),CUresult,(CUarray,Csize_t,CUarray,Csize_t,Csize_t),dstArray,dstOffset,srcArray,srcOffset,ByteCount))
end

function cuMemcpy2D_v2(pCopy)
    check_curesult(ccall((:cuMemcpy2D_v2,libcuda),CUresult,(Ptr{CUDA_MEMCPY2D},),pCopy))
end

function cuMemcpy2DUnaligned_v2(pCopy)
    check_curesult(ccall((:cuMemcpy2DUnaligned_v2,libcuda),CUresult,(Ptr{CUDA_MEMCPY2D},),pCopy))
end

function cuMemcpy3D_v2(pCopy)
    check_curesult(ccall((:cuMemcpy3D_v2,libcuda),CUresult,(Ptr{CUDA_MEMCPY3D},),pCopy))
end

function cuMemcpy3DPeer(pCopy)
    check_curesult(ccall((:cuMemcpy3DPeer,libcuda),CUresult,(Ptr{CUDA_MEMCPY3D_PEER},),pCopy))
end

function cuMemcpyAsync(dst,src,ByteCount,hStream)
    check_curesult(ccall((:cuMemcpyAsync,libcuda),CUresult,(CUdeviceptr,CUdeviceptr,Csize_t,CUstream),dst,src,ByteCount,hStream))
end

function cuMemcpyPeerAsync(dstDevice,dstContext,srcDevice,srcContext,ByteCount,hStream)
    check_curesult(ccall((:cuMemcpyPeerAsync,libcuda),CUresult,(CUdeviceptr,CUcontext,CUdeviceptr,CUcontext,Csize_t,CUstream),dstDevice,dstContext,srcDevice,srcContext,ByteCount,hStream))
end

function cuMemcpyHtoDAsync_v2(dstDevice,srcHost,ByteCount,hStream)
    check_curesult(ccall((:cuMemcpyHtoDAsync_v2,libcuda),CUresult,(CUdeviceptr,Ptr{Void},Csize_t,CUstream),dstDevice,srcHost,ByteCount,hStream))
end

function cuMemcpyDtoHAsync_v2(dstHost,srcDevice,ByteCount,hStream)
    check_curesult(ccall((:cuMemcpyDtoHAsync_v2,libcuda),CUresult,(Ptr{Void},CUdeviceptr,Csize_t,CUstream),dstHost,srcDevice,ByteCount,hStream))
end

function cuMemcpyDtoDAsync_v2(dstDevice,srcDevice,ByteCount,hStream)
    check_curesult(ccall((:cuMemcpyDtoDAsync_v2,libcuda),CUresult,(CUdeviceptr,CUdeviceptr,Csize_t,CUstream),dstDevice,srcDevice,ByteCount,hStream))
end

function cuMemcpyHtoAAsync_v2(dstArray,dstOffset,srcHost,ByteCount,hStream)
    check_curesult(ccall((:cuMemcpyHtoAAsync_v2,libcuda),CUresult,(CUarray,Csize_t,Ptr{Void},Csize_t,CUstream),dstArray,dstOffset,srcHost,ByteCount,hStream))
end

function cuMemcpyAtoHAsync_v2(dstHost,srcArray,srcOffset,ByteCount,hStream)
    check_curesult(ccall((:cuMemcpyAtoHAsync_v2,libcuda),CUresult,(Ptr{Void},CUarray,Csize_t,Csize_t,CUstream),dstHost,srcArray,srcOffset,ByteCount,hStream))
end

function cuMemcpy2DAsync_v2(pCopy,hStream)
    check_curesult(ccall((:cuMemcpy2DAsync_v2,libcuda),CUresult,(Ptr{CUDA_MEMCPY2D},CUstream),pCopy,hStream))
end

function cuMemcpy3DAsync_v2(pCopy,hStream)
    check_curesult(ccall((:cuMemcpy3DAsync_v2,libcuda),CUresult,(Ptr{CUDA_MEMCPY3D},CUstream),pCopy,hStream))
end

function cuMemcpy3DPeerAsync(pCopy,hStream)
    check_curesult(ccall((:cuMemcpy3DPeerAsync,libcuda),CUresult,(Ptr{CUDA_MEMCPY3D_PEER},CUstream),pCopy,hStream))
end

function cuMemsetD8_v2(dstDevice,uc,N)
    check_curesult(ccall((:cuMemsetD8_v2,libcuda),CUresult,(CUdeviceptr,Cuchar,Csize_t),dstDevice,uc,N))
end

function cuMemsetD16_v2(dstDevice,us,N)
    check_curesult(ccall((:cuMemsetD16_v2,libcuda),CUresult,(CUdeviceptr,UInt16,Csize_t),dstDevice,us,N))
end

function cuMemsetD32_v2(dstDevice,ui,N)
    check_curesult(ccall((:cuMemsetD32_v2,libcuda),CUresult,(CUdeviceptr,UInt32,Csize_t),dstDevice,ui,N))
end

function cuMemsetD2D8_v2(dstDevice,dstPitch,uc,Width,Height)
    check_curesult(ccall((:cuMemsetD2D8_v2,libcuda),CUresult,(CUdeviceptr,Csize_t,Cuchar,Csize_t,Csize_t),dstDevice,dstPitch,uc,Width,Height))
end

function cuMemsetD2D16_v2(dstDevice,dstPitch,us,Width,Height)
    check_curesult(ccall((:cuMemsetD2D16_v2,libcuda),CUresult,(CUdeviceptr,Csize_t,UInt16,Csize_t,Csize_t),dstDevice,dstPitch,us,Width,Height))
end

function cuMemsetD2D32_v2(dstDevice,dstPitch,ui,Width,Height)
    check_curesult(ccall((:cuMemsetD2D32_v2,libcuda),CUresult,(CUdeviceptr,Csize_t,UInt32,Csize_t,Csize_t),dstDevice,dstPitch,ui,Width,Height))
end

function cuMemsetD8Async(dstDevice,uc,N,hStream)
    check_curesult(ccall((:cuMemsetD8Async,libcuda),CUresult,(CUdeviceptr,Cuchar,Csize_t,CUstream),dstDevice,uc,N,hStream))
end

function cuMemsetD16Async(dstDevice,us,N,hStream)
    check_curesult(ccall((:cuMemsetD16Async,libcuda),CUresult,(CUdeviceptr,UInt16,Csize_t,CUstream),dstDevice,us,N,hStream))
end

function cuMemsetD32Async(dstDevice,ui,N,hStream)
    check_curesult(ccall((:cuMemsetD32Async,libcuda),CUresult,(CUdeviceptr,UInt32,Csize_t,CUstream),dstDevice,ui,N,hStream))
end

function cuMemsetD2D8Async(dstDevice,dstPitch,uc,Width,Height,hStream)
    check_curesult(ccall((:cuMemsetD2D8Async,libcuda),CUresult,(CUdeviceptr,Csize_t,Cuchar,Csize_t,Csize_t,CUstream),dstDevice,dstPitch,uc,Width,Height,hStream))
end

function cuMemsetD2D16Async(dstDevice,dstPitch,us,Width,Height,hStream)
    check_curesult(ccall((:cuMemsetD2D16Async,libcuda),CUresult,(CUdeviceptr,Csize_t,UInt16,Csize_t,Csize_t,CUstream),dstDevice,dstPitch,us,Width,Height,hStream))
end

function cuMemsetD2D32Async(dstDevice,dstPitch,ui,Width,Height,hStream)
    check_curesult(ccall((:cuMemsetD2D32Async,libcuda),CUresult,(CUdeviceptr,Csize_t,UInt32,Csize_t,Csize_t,CUstream),dstDevice,dstPitch,ui,Width,Height,hStream))
end

function cuArrayCreate_v2(pHandle,pAllocateArray)
    check_curesult(ccall((:cuArrayCreate_v2,libcuda),CUresult,(Ptr{CUarray},Ptr{CUDA_ARRAY_DESCRIPTOR}),pHandle,pAllocateArray))
end

function cuArrayGetDescriptor_v2(pArrayDescriptor,hArray)
    check_curesult(ccall((:cuArrayGetDescriptor_v2,libcuda),CUresult,(Ptr{CUDA_ARRAY_DESCRIPTOR},CUarray),pArrayDescriptor,hArray))
end

function cuArrayDestroy(hArray)
    check_curesult(ccall((:cuArrayDestroy,libcuda),CUresult,(CUarray,),hArray))
end

function cuArray3DCreate_v2(pHandle,pAllocateArray)
    check_curesult(ccall((:cuArray3DCreate_v2,libcuda),CUresult,(Ptr{CUarray},Ptr{CUDA_ARRAY3D_DESCRIPTOR}),pHandle,pAllocateArray))
end

function cuArray3DGetDescriptor_v2(pArrayDescriptor,hArray)
    check_curesult(ccall((:cuArray3DGetDescriptor_v2,libcuda),CUresult,(Ptr{CUDA_ARRAY3D_DESCRIPTOR},CUarray),pArrayDescriptor,hArray))
end

function cuMipmappedArrayCreate(pHandle,pMipmappedArrayDesc,numMipmapLevels)
    check_curesult(ccall((:cuMipmappedArrayCreate,libcuda),CUresult,(Ptr{CUmipmappedArray},Ptr{CUDA_ARRAY3D_DESCRIPTOR},UInt32),pHandle,pMipmappedArrayDesc,numMipmapLevels))
end

function cuMipmappedArrayGetLevel(pLevelArray,hMipmappedArray,level)
    check_curesult(ccall((:cuMipmappedArrayGetLevel,libcuda),CUresult,(Ptr{CUarray},CUmipmappedArray,UInt32),pLevelArray,hMipmappedArray,level))
end

function cuMipmappedArrayDestroy(hMipmappedArray)
    check_curesult(ccall((:cuMipmappedArrayDestroy,libcuda),CUresult,(CUmipmappedArray,),hMipmappedArray))
end

function cuPointerGetAttribute(data,attribute,ptr)
    check_curesult(ccall((:cuPointerGetAttribute,libcuda),CUresult,(Ptr{Void},CUpointer_attribute,CUdeviceptr),data,attribute,ptr))
end

function cuPointerSetAttribute(value,attribute,ptr)
    check_curesult(ccall((:cuPointerSetAttribute,libcuda),CUresult,(Ptr{Void},CUpointer_attribute,CUdeviceptr),value,attribute,ptr))
end

function cuPointerGetAttributes(numAttributes,attributes,data,ptr)
    check_curesult(ccall((:cuPointerGetAttributes,libcuda),CUresult,(UInt32,Ptr{CUpointer_attribute},Ptr{Ptr{Void}},CUdeviceptr),numAttributes,attributes,data,ptr))
end

function cuStreamCreate(phStream,Flags)
    check_curesult(ccall((:cuStreamCreate,libcuda),CUresult,(Ptr{CUstream},UInt32),phStream,Flags))
end

function cuStreamCreateWithPriority(phStream,flags,priority)
    check_curesult(ccall((:cuStreamCreateWithPriority,libcuda),CUresult,(Ptr{CUstream},UInt32,Cint),phStream,flags,priority))
end

function cuStreamGetPriority(hStream,priority)
    check_curesult(ccall((:cuStreamGetPriority,libcuda),CUresult,(CUstream,Ptr{Cint}),hStream,priority))
end

function cuStreamGetFlags(hStream,flags)
    check_curesult(ccall((:cuStreamGetFlags,libcuda),CUresult,(CUstream,Ptr{UInt32}),hStream,flags))
end

function cuStreamWaitEvent(hStream,hEvent,Flags)
    check_curesult(ccall((:cuStreamWaitEvent,libcuda),CUresult,(CUstream,CUevent,UInt32),hStream,hEvent,Flags))
end

function cuStreamAddCallback(hStream,callback,userData,flags)
    check_curesult(ccall((:cuStreamAddCallback,libcuda),CUresult,(CUstream,CUstreamCallback,Ptr{Void},UInt32),hStream,callback,userData,flags))
end

function cuStreamAttachMemAsync(hStream,dptr,length,flags)
    check_curesult(ccall((:cuStreamAttachMemAsync,libcuda),CUresult,(CUstream,CUdeviceptr,Csize_t,UInt32),hStream,dptr,length,flags))
end

function cuStreamQuery(hStream)
    check_curesult(ccall((:cuStreamQuery,libcuda),CUresult,(CUstream,),hStream))
end

function cuStreamSynchronize(hStream)
    check_curesult(ccall((:cuStreamSynchronize,libcuda),CUresult,(CUstream,),hStream))
end

function cuStreamDestroy_v2(hStream)
    check_curesult(ccall((:cuStreamDestroy_v2,libcuda),CUresult,(CUstream,),hStream))
end

function cuEventCreate(phEvent,Flags)
    check_curesult(ccall((:cuEventCreate,libcuda),CUresult,(Ptr{CUevent},UInt32),phEvent,Flags))
end

function cuEventRecord(hEvent,hStream)
    check_curesult(ccall((:cuEventRecord,libcuda),CUresult,(CUevent,CUstream),hEvent,hStream))
end

function cuEventQuery(hEvent)
    check_curesult(ccall((:cuEventQuery,libcuda),CUresult,(CUevent,),hEvent))
end

function cuEventSynchronize(hEvent)
    check_curesult(ccall((:cuEventSynchronize,libcuda),CUresult,(CUevent,),hEvent))
end

function cuEventDestroy_v2(hEvent)
    check_curesult(ccall((:cuEventDestroy_v2,libcuda),CUresult,(CUevent,),hEvent))
end

function cuEventElapsedTime(pMilliseconds,hStart,hEnd)
    check_curesult(ccall((:cuEventElapsedTime,libcuda),CUresult,(Ptr{Cfloat},CUevent,CUevent),pMilliseconds,hStart,hEnd))
end

function cuFuncGetAttribute(pi,attrib,hfunc)
    check_curesult(ccall((:cuFuncGetAttribute,libcuda),CUresult,(Ptr{Cint},CUfunction_attribute,CUfunction),pi,attrib,hfunc))
end

function cuFuncSetCacheConfig(hfunc,config)
    check_curesult(ccall((:cuFuncSetCacheConfig,libcuda),CUresult,(CUfunction,CUfunc_cache),hfunc,config))
end

function cuFuncSetSharedMemConfig(hfunc,config)
    check_curesult(ccall((:cuFuncSetSharedMemConfig,libcuda),CUresult,(CUfunction,CUsharedconfig),hfunc,config))
end

function cuLaunchKernel(f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,hStream,kernelParams,extra)
    check_curesult(ccall((:cuLaunchKernel,libcuda),CUresult,(CUfunction,UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,CUstream,Ptr{Ptr{Void}},Ptr{Ptr{Void}}),f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,hStream,kernelParams,extra))
end

function cuFuncSetBlockShape(hfunc,x,y,z)
    check_curesult(ccall((:cuFuncSetBlockShape,libcuda),CUresult,(CUfunction,Cint,Cint,Cint),hfunc,x,y,z))
end

function cuFuncSetSharedSize(hfunc,bytes)
    check_curesult(ccall((:cuFuncSetSharedSize,libcuda),CUresult,(CUfunction,UInt32),hfunc,bytes))
end

function cuParamSetSize(hfunc,numbytes)
    check_curesult(ccall((:cuParamSetSize,libcuda),CUresult,(CUfunction,UInt32),hfunc,numbytes))
end

function cuParamSeti(hfunc,offset,value)
    check_curesult(ccall((:cuParamSeti,libcuda),CUresult,(CUfunction,Cint,UInt32),hfunc,offset,value))
end

function cuParamSetf(hfunc,offset,value)
    check_curesult(ccall((:cuParamSetf,libcuda),CUresult,(CUfunction,Cint,Cfloat),hfunc,offset,value))
end

function cuParamSetv(hfunc,offset,ptr,numbytes)
    check_curesult(ccall((:cuParamSetv,libcuda),CUresult,(CUfunction,Cint,Ptr{Void},UInt32),hfunc,offset,ptr,numbytes))
end

function cuLaunch(f)
    check_curesult(ccall((:cuLaunch,libcuda),CUresult,(CUfunction,),f))
end

function cuLaunchGrid(f,grid_width,grid_height)
    check_curesult(ccall((:cuLaunchGrid,libcuda),CUresult,(CUfunction,Cint,Cint),f,grid_width,grid_height))
end

function cuLaunchGridAsync(f,grid_width,grid_height,hStream)
    check_curesult(ccall((:cuLaunchGridAsync,libcuda),CUresult,(CUfunction,Cint,Cint,CUstream),f,grid_width,grid_height,hStream))
end

function cuParamSetTexRef(hfunc,texunit,hTexRef)
    check_curesult(ccall((:cuParamSetTexRef,libcuda),CUresult,(CUfunction,Cint,CUtexref),hfunc,texunit,hTexRef))
end

function cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks,func,blockSize,dynamicSMemSize)
    check_curesult(ccall((:cuOccupancyMaxActiveBlocksPerMultiprocessor,libcuda),CUresult,(Ptr{Cint},CUfunction,Cint,Csize_t),numBlocks,func,blockSize,dynamicSMemSize))
end

function cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks,func,blockSize,dynamicSMemSize,flags)
    check_curesult(ccall((:cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,libcuda),CUresult,(Ptr{Cint},CUfunction,Cint,Csize_t,UInt32),numBlocks,func,blockSize,dynamicSMemSize,flags))
end

function cuOccupancyMaxPotentialBlockSize(minGridSize,blockSize,func,blockSizeToDynamicSMemSize,dynamicSMemSize,blockSizeLimit)
    check_curesult(ccall((:cuOccupancyMaxPotentialBlockSize,libcuda),CUresult,(Ptr{Cint},Ptr{Cint},CUfunction,CUoccupancyB2DSize,Csize_t,Cint),minGridSize,blockSize,func,blockSizeToDynamicSMemSize,dynamicSMemSize,blockSizeLimit))
end

function cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize,blockSize,func,blockSizeToDynamicSMemSize,dynamicSMemSize,blockSizeLimit,flags)
    check_curesult(ccall((:cuOccupancyMaxPotentialBlockSizeWithFlags,libcuda),CUresult,(Ptr{Cint},Ptr{Cint},CUfunction,CUoccupancyB2DSize,Csize_t,Cint,UInt32),minGridSize,blockSize,func,blockSizeToDynamicSMemSize,dynamicSMemSize,blockSizeLimit,flags))
end

function cuTexRefSetArray(hTexRef,hArray,Flags)
    check_curesult(ccall((:cuTexRefSetArray,libcuda),CUresult,(CUtexref,CUarray,UInt32),hTexRef,hArray,Flags))
end

function cuTexRefSetMipmappedArray(hTexRef,hMipmappedArray,Flags)
    check_curesult(ccall((:cuTexRefSetMipmappedArray,libcuda),CUresult,(CUtexref,CUmipmappedArray,UInt32),hTexRef,hMipmappedArray,Flags))
end

function cuTexRefSetAddress_v2(ByteOffset,hTexRef,dptr,bytes)
    check_curesult(ccall((:cuTexRefSetAddress_v2,libcuda),CUresult,(Ptr{Csize_t},CUtexref,CUdeviceptr,Csize_t),ByteOffset,hTexRef,dptr,bytes))
end

function cuTexRefSetAddress2D_v3(hTexRef,desc,dptr,Pitch)
    check_curesult(ccall((:cuTexRefSetAddress2D_v3,libcuda),CUresult,(CUtexref,Ptr{CUDA_ARRAY_DESCRIPTOR},CUdeviceptr,Csize_t),hTexRef,desc,dptr,Pitch))
end

function cuTexRefSetFormat(hTexRef,fmt,NumPackedComponents)
    check_curesult(ccall((:cuTexRefSetFormat,libcuda),CUresult,(CUtexref,CUarray_format,Cint),hTexRef,fmt,NumPackedComponents))
end

function cuTexRefSetAddressMode(hTexRef,dim,am)
    check_curesult(ccall((:cuTexRefSetAddressMode,libcuda),CUresult,(CUtexref,Cint,CUaddress_mode),hTexRef,dim,am))
end

function cuTexRefSetFilterMode(hTexRef,fm)
    check_curesult(ccall((:cuTexRefSetFilterMode,libcuda),CUresult,(CUtexref,CUfilter_mode),hTexRef,fm))
end

function cuTexRefSetMipmapFilterMode(hTexRef,fm)
    check_curesult(ccall((:cuTexRefSetMipmapFilterMode,libcuda),CUresult,(CUtexref,CUfilter_mode),hTexRef,fm))
end

function cuTexRefSetMipmapLevelBias(hTexRef,bias)
    check_curesult(ccall((:cuTexRefSetMipmapLevelBias,libcuda),CUresult,(CUtexref,Cfloat),hTexRef,bias))
end

function cuTexRefSetMipmapLevelClamp(hTexRef,minMipmapLevelClamp,maxMipmapLevelClamp)
    check_curesult(ccall((:cuTexRefSetMipmapLevelClamp,libcuda),CUresult,(CUtexref,Cfloat,Cfloat),hTexRef,minMipmapLevelClamp,maxMipmapLevelClamp))
end

function cuTexRefSetMaxAnisotropy(hTexRef,maxAniso)
    check_curesult(ccall((:cuTexRefSetMaxAnisotropy,libcuda),CUresult,(CUtexref,UInt32),hTexRef,maxAniso))
end

function cuTexRefSetFlags(hTexRef,Flags)
    check_curesult(ccall((:cuTexRefSetFlags,libcuda),CUresult,(CUtexref,UInt32),hTexRef,Flags))
end

function cuTexRefGetAddress_v2(pdptr,hTexRef)
    check_curesult(ccall((:cuTexRefGetAddress_v2,libcuda),CUresult,(Ptr{CUdeviceptr},CUtexref),pdptr,hTexRef))
end

function cuTexRefGetArray(phArray,hTexRef)
    check_curesult(ccall((:cuTexRefGetArray,libcuda),CUresult,(Ptr{CUarray},CUtexref),phArray,hTexRef))
end

function cuTexRefGetMipmappedArray(phMipmappedArray,hTexRef)
    check_curesult(ccall((:cuTexRefGetMipmappedArray,libcuda),CUresult,(Ptr{CUmipmappedArray},CUtexref),phMipmappedArray,hTexRef))
end

function cuTexRefGetAddressMode(pam,hTexRef,dim)
    check_curesult(ccall((:cuTexRefGetAddressMode,libcuda),CUresult,(Ptr{CUaddress_mode},CUtexref,Cint),pam,hTexRef,dim))
end

function cuTexRefGetFilterMode(pfm,hTexRef)
    check_curesult(ccall((:cuTexRefGetFilterMode,libcuda),CUresult,(Ptr{CUfilter_mode},CUtexref),pfm,hTexRef))
end

function cuTexRefGetFormat(pFormat,pNumChannels,hTexRef)
    check_curesult(ccall((:cuTexRefGetFormat,libcuda),CUresult,(Ptr{CUarray_format},Ptr{Cint},CUtexref),pFormat,pNumChannels,hTexRef))
end

function cuTexRefGetMipmapFilterMode(pfm,hTexRef)
    check_curesult(ccall((:cuTexRefGetMipmapFilterMode,libcuda),CUresult,(Ptr{CUfilter_mode},CUtexref),pfm,hTexRef))
end

function cuTexRefGetMipmapLevelBias(pbias,hTexRef)
    check_curesult(ccall((:cuTexRefGetMipmapLevelBias,libcuda),CUresult,(Ptr{Cfloat},CUtexref),pbias,hTexRef))
end

function cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp,pmaxMipmapLevelClamp,hTexRef)
    check_curesult(ccall((:cuTexRefGetMipmapLevelClamp,libcuda),CUresult,(Ptr{Cfloat},Ptr{Cfloat},CUtexref),pminMipmapLevelClamp,pmaxMipmapLevelClamp,hTexRef))
end

function cuTexRefGetMaxAnisotropy(pmaxAniso,hTexRef)
    check_curesult(ccall((:cuTexRefGetMaxAnisotropy,libcuda),CUresult,(Ptr{Cint},CUtexref),pmaxAniso,hTexRef))
end

function cuTexRefGetFlags(pFlags,hTexRef)
    check_curesult(ccall((:cuTexRefGetFlags,libcuda),CUresult,(Ptr{UInt32},CUtexref),pFlags,hTexRef))
end

function cuTexRefCreate(pTexRef)
    check_curesult(ccall((:cuTexRefCreate,libcuda),CUresult,(Ptr{CUtexref},),pTexRef))
end

function cuTexRefDestroy(hTexRef)
    check_curesult(ccall((:cuTexRefDestroy,libcuda),CUresult,(CUtexref,),hTexRef))
end

function cuSurfRefSetArray(hSurfRef,hArray,Flags)
    check_curesult(ccall((:cuSurfRefSetArray,libcuda),CUresult,(CUsurfref,CUarray,UInt32),hSurfRef,hArray,Flags))
end

function cuSurfRefGetArray(phArray,hSurfRef)
    check_curesult(ccall((:cuSurfRefGetArray,libcuda),CUresult,(Ptr{CUarray},CUsurfref),phArray,hSurfRef))
end

function cuTexObjectCreate(pTexObject,pResDesc,pTexDesc,pResViewDesc)
    check_curesult(ccall((:cuTexObjectCreate,libcuda),CUresult,(Ptr{CUtexObject},Ptr{CUDA_RESOURCE_DESC},Ptr{CUDA_TEXTURE_DESC},Ptr{CUDA_RESOURCE_VIEW_DESC}),pTexObject,pResDesc,pTexDesc,pResViewDesc))
end

function cuTexObjectDestroy(texObject)
    check_curesult(ccall((:cuTexObjectDestroy,libcuda),CUresult,(CUtexObject,),texObject))
end

function cuTexObjectGetResourceDesc(pResDesc,texObject)
    check_curesult(ccall((:cuTexObjectGetResourceDesc,libcuda),CUresult,(Ptr{CUDA_RESOURCE_DESC},CUtexObject),pResDesc,texObject))
end

function cuTexObjectGetTextureDesc(pTexDesc,texObject)
    check_curesult(ccall((:cuTexObjectGetTextureDesc,libcuda),CUresult,(Ptr{CUDA_TEXTURE_DESC},CUtexObject),pTexDesc,texObject))
end

function cuTexObjectGetResourceViewDesc(pResViewDesc,texObject)
    check_curesult(ccall((:cuTexObjectGetResourceViewDesc,libcuda),CUresult,(Ptr{CUDA_RESOURCE_VIEW_DESC},CUtexObject),pResViewDesc,texObject))
end

function cuSurfObjectCreate(pSurfObject,pResDesc)
    check_curesult(ccall((:cuSurfObjectCreate,libcuda),CUresult,(Ptr{CUsurfObject},Ptr{CUDA_RESOURCE_DESC}),pSurfObject,pResDesc))
end

function cuSurfObjectDestroy(surfObject)
    check_curesult(ccall((:cuSurfObjectDestroy,libcuda),CUresult,(CUsurfObject,),surfObject))
end

function cuSurfObjectGetResourceDesc(pResDesc,surfObject)
    check_curesult(ccall((:cuSurfObjectGetResourceDesc,libcuda),CUresult,(Ptr{CUDA_RESOURCE_DESC},CUsurfObject),pResDesc,surfObject))
end

function cuDeviceCanAccessPeer(canAccessPeer,dev,peerDev)
    check_curesult(ccall((:cuDeviceCanAccessPeer,libcuda),CUresult,(Ptr{Cint},CUdevice,CUdevice),canAccessPeer,dev,peerDev))
end

function cuCtxEnablePeerAccess(peerContext,Flags)
    check_curesult(ccall((:cuCtxEnablePeerAccess,libcuda),CUresult,(CUcontext,UInt32),peerContext,Flags))
end

function cuCtxDisablePeerAccess(peerContext)
    check_curesult(ccall((:cuCtxDisablePeerAccess,libcuda),CUresult,(CUcontext,),peerContext))
end

function cuGraphicsUnregisterResource(resource)
    check_curesult(ccall((:cuGraphicsUnregisterResource,libcuda),CUresult,(CUgraphicsResource,),resource))
end

function cuGraphicsSubResourceGetMappedArray(pArray,resource,arrayIndex,mipLevel)
    check_curesult(ccall((:cuGraphicsSubResourceGetMappedArray,libcuda),CUresult,(Ptr{CUarray},CUgraphicsResource,UInt32,UInt32),pArray,resource,arrayIndex,mipLevel))
end

function cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray,resource)
    check_curesult(ccall((:cuGraphicsResourceGetMappedMipmappedArray,libcuda),CUresult,(Ptr{CUmipmappedArray},CUgraphicsResource),pMipmappedArray,resource))
end

function cuGraphicsResourceGetMappedPointer_v2(pDevPtr,pSize,resource)
    check_curesult(ccall((:cuGraphicsResourceGetMappedPointer_v2,libcuda),CUresult,(Ptr{CUdeviceptr},Ptr{Csize_t},CUgraphicsResource),pDevPtr,pSize,resource))
end

function cuGraphicsResourceSetMapFlags_v2(resource,flags)
    check_curesult(ccall((:cuGraphicsResourceSetMapFlags_v2,libcuda),CUresult,(CUgraphicsResource,UInt32),resource,flags))
end

function cuGraphicsMapResources(count,resources,hStream)
    check_curesult(ccall((:cuGraphicsMapResources,libcuda),CUresult,(UInt32,Ptr{CUgraphicsResource},CUstream),count,resources,hStream))
end

function cuGraphicsUnmapResources(count,resources,hStream)
    check_curesult(ccall((:cuGraphicsUnmapResources,libcuda),CUresult,(UInt32,Ptr{CUgraphicsResource},CUstream),count,resources,hStream))
end

function cuGetExportTable(ppExportTable,pExportTableId)
    check_curesult(ccall((:cuGetExportTable,libcuda),CUresult,(Ptr{Ptr{Void}},Ptr{CUuuid}),ppExportTable,pExportTableId))
end
S_WRITE_DISCARD = (UInt32)(2)
# end enum CUgraphicsMapResourceFlags

# begin enum CUarray_cubemap_face_enum
const CUarray_cubemap_face_enum = UInt32
const CU_CUBEMAP_FACE_POSITIVE_X = (UInt32)(0)
const CU_CUBEMAP_FACE_NEGATIVE_X = (UInt32)(1)
const CU_CUBEMAP_FACE_POSITIVE_Y = (UInt32)(2)
const CU_CUBEMAP_FACE_NEGATIVE_Y = (UInt32)(3)
const CU_CUBEMAP_FACE_POSITIVE_Z = (UInt32)(4)
const CU_CUBEMAP_FACE_NEGATIVE_Z = (UInt32)(5)
# end enum CUarray_cubemap_face_enum

# begin enum CUarray_cubemap_face
const CUarray_cubemap_face = UInt32
const CU_CUBEMAP_FACE_POSITIVE_X = (UInt32)(0)
const CU_CUBEMAP_FACE_NEGATIVE_X = (UInt32)(1)
const CU_CUBEMAP_FACE_POSITIVE_Y = (UInt32)(2)
const CU_CUBEMAP_FACE_NEGATIVE_Y = (UInt32)(3)
const CU_CUBEMAP_FACE_POSITIVE_Z = (UInt32)(4)
const CU_CUBEMAP_FACE_NEGATIVE_Z = (UInt32)(5)
# end enum CUarray_cubemap_face

# begin enum CUlimit_enum
const CUlimit_enum = UInt32
const CU_LIMIT_STACK_SIZE = (UInt32)(0)
const CU_LIMIT_PRINTF_FIFO_SIZE = (UInt32)(1)
const CU_LIMIT_MALLOC_HEAP_SIZE = (UInt32)(2)
const CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = (UInt32)(3)
const CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = (UInt32)(4)
const CU_LIMIT_MAX = (UInt32)(5)
# end enum CUlimit_enum

# begin enum CUlimit
const CUlimit = UInt32
const CU_LIMIT_STACK_SIZE = (UInt32)(0)
const CU_LIMIT_PRINTF_FIFO_SIZE = (UInt32)(1)
const CU_LIMIT_MALLOC_HEAP_SIZE = (UInt32)(2)
const CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = (UInt32)(3)
const CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = (UInt32)(4)
const CU_LIMIT_MAX = (UInt32)(5)
# end enum CUlimit

# begin enum CUresourcetype_enum
const CUresourcetype_enum = UInt32
const CU_RESOURCE_TYPE_ARRAY = (UInt32)(0)
const CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = (UInt32)(1)
const CU_RESOURCE_TYPE_LINEAR = (UInt32)(2)
const CU_RESOURCE_TYPE_PITCH2D = (UInt32)(3)
# end enum CUresourcetype_enum

# begin enum CUresourcetype
const CUresourcetype = UInt32
const CU_RESOURCE_TYPE_ARRAY = (UInt32)(0)
const CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = (UInt32)(1)
const CU_RESOURCE_TYPE_LINEAR = (UInt32)(2)
const CU_RESOURCE_TYPE_PITCH2D = (UInt32)(3)
# end enum CUresourcetype

# begin enum cudaError_enum
const cudaError_enum = UInt32
const CUDA_SUCCESS = (UInt32)(0)
const CUDA_ERROR_INVALID_VALUE = (UInt32)(1)
const CUDA_ERROR_OUT_OF_MEMORY = (UInt32)(2)
const CUDA_ERROR_NOT_INITIALIZED = (UInt32)(3)
const CUDA_ERROR_DEINITIALIZED = (UInt32)(4)
const CUDA_ERROR_PROFILER_DISABLED = (UInt32)(5)
const CUDA_ERROR_PROFILER_NOT_INITIALIZED = (UInt32)(6)
const CUDA_ERROR_PROFILER_ALREADY_STARTED = (UInt32)(7)
const CUDA_ERROR_PROFILER_ALREADY_STOPPED = (UInt32)(8)
const CUDA_ERROR_NO_DEVICE = (UInt32)(100)
const CUDA_ERROR_INVALID_DEVICE = (UInt32)(101)
const CUDA_ERROR_INVALID_IMAGE = (UInt32)(200)
const CUDA_ERROR_INVALID_CONTEXT = (UInt32)(201)
const CUDA_ERROR_CONTEXT_ALREADY_CURRENT = (UInt32)(202)
const CUDA_ERROR_MAP_FAILED = (UInt32)(205)
const CUDA_ERROR_UNMAP_FAILED = (UInt32)(206)
const CUDA_ERROR_ARRAY_IS_MAPPED = (UInt32)(207)
const CUDA_ERROR_ALREADY_MAPPED = (UInt32)(208)
const CUDA_ERROR_NO_BINARY_FOR_GPU = (UInt32)(209)
const CUDA_ERROR_ALREADY_ACQUIRED = (UInt32)(210)
const CUDA_ERROR_NOT_MAPPED = (UInt32)(211)
const CUDA_ERROR_NOT_MAPPED_AS_ARRAY = (UInt32)(212)
const CUDA_ERROR_NOT_MAPPED_AS_POINTER = (UInt32)(213)
const CUDA_ERROR_ECC_UNCORRECTABLE = (UInt32)(214)
const CUDA_ERROR_UNSUPPORTED_LIMIT = (UInt32)(215)
const CUDA_ERROR_CONTEXT_ALREADY_IN_USE = (UInt32)(216)
const CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = (UInt32)(217)
const CUDA_ERROR_INVALID_PTX = (UInt32)(218)
const CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = (UInt32)(219)
const CUDA_ERROR_INVALID_SOURCE = (UInt32)(300)
const CUDA_ERROR_FILE_NOT_FOUND = (UInt32)(301)
const CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = (UInt32)(302)
const CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = (UInt32)(303)
const CUDA_ERROR_OPERATING_SYSTEM = (UInt32)(304)
const CUDA_ERROR_INVALID_HANDLE = (UInt32)(400)
const CUDA_ERROR_NOT_FOUND = (UInt32)(500)
const CUDA_ERROR_NOT_READY = (UInt32)(600)
const CUDA_ERROR_ILLEGAL_ADDRESS = (UInt32)(700)
const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = (UInt32)(701)
const CUDA_ERROR_LAUNCH_TIMEOUT = (UInt32)(702)
const CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = (UInt32)(703)
const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = (UInt32)(704)
const CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = (UInt32)(705)
const CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = (UInt32)(708)
const CUDA_ERROR_CONTEXT_IS_DESTROYED = (UInt32)(709)
const CUDA_ERROR_ASSERT = (UInt32)(710)
const CUDA_ERROR_TOO_MANY_PEERS = (UInt32)(711)
const CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = (UInt32)(712)
const CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = (UInt32)(713)
const CUDA_ERROR_HARDWARE_STACK_ERROR = (UInt32)(714)
const CUDA_ERROR_ILLEGAL_INSTRUCTION = (UInt32)(715)
const CUDA_ERROR_MISALIGNED_ADDRESS = (UInt32)(716)
const CUDA_ERROR_INVALID_ADDRESS_SPACE = (UInt32)(717)
const CUDA_ERROR_INVALID_PC = (UInt32)(718)
const CUDA_ERROR_LAUNCH_FAILED = (UInt32)(719)
const CUDA_ERROR_NOT_PERMITTED = (UInt32)(800)
const CUDA_ERROR_NOT_SUPPORTED = (UInt32)(801)
const CUDA_ERROR_UNKNOWN = (UInt32)(999)
# end enum cudaError_enum

# begin enum CUresult
const CUresult = UInt32
const CUDA_SUCCESS = (UInt32)(0)
const CUDA_ERROR_INVALID_VALUE = (UInt32)(1)
const CUDA_ERROR_OUT_OF_MEMORY = (UInt32)(2)
const CUDA_ERROR_NOT_INITIALIZED = (UInt32)(3)
const CUDA_ERROR_DEINITIALIZED = (UInt32)(4)
const CUDA_ERROR_PROFILER_DISABLED = (UInt32)(5)
const CUDA_ERROR_PROFILER_NOT_INITIALIZED = (UInt32)(6)
const CUDA_ERROR_PROFILER_ALREADY_STARTED = (UInt32)(7)
const CUDA_ERROR_PROFILER_ALREADY_STOPPED = (UInt32)(8)
const CUDA_ERROR_NO_DEVICE = (UInt32)(100)
const CUDA_ERROR_INVALID_DEVICE = (UInt32)(101)
const CUDA_ERROR_INVALID_IMAGE = (UInt32)(200)
const CUDA_ERROR_INVALID_CONTEXT = (UInt32)(201)
const CUDA_ERROR_CONTEXT_ALREADY_CURRENT = (UInt32)(202)
const CUDA_ERROR_MAP_FAILED = (UInt32)(205)
const CUDA_ERROR_UNMAP_FAILED = (UInt32)(206)
const CUDA_ERROR_ARRAY_IS_MAPPED = (UInt32)(207)
const CUDA_ERROR_ALREADY_MAPPED = (UInt32)(208)
const CUDA_ERROR_NO_BINARY_FOR_GPU = (UInt32)(209)
const CUDA_ERROR_ALREADY_ACQUIRED = (UInt32)(210)
const CUDA_ERROR_NOT_MAPPED = (UInt32)(211)
const CUDA_ERROR_NOT_MAPPED_AS_ARRAY = (UInt32)(212)
const CUDA_ERROR_NOT_MAPPED_AS_POINTER = (UInt32)(213)
const CUDA_ERROR_ECC_UNCORRECTABLE = (UInt32)(214)
const CUDA_ERROR_UNSUPPORTED_LIMIT = (UInt32)(215)
const CUDA_ERROR_CONTEXT_ALREADY_IN_USE = (UInt32)(216)
const CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = (UInt32)(217)
const CUDA_ERROR_INVALID_PTX = (UInt32)(218)
const CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = (UInt32)(219)
const CUDA_ERROR_INVALID_SOURCE = (UInt32)(300)
const CUDA_ERROR_FILE_NOT_FOUND = (UInt32)(301)
const CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = (UInt32)(302)
const CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = (UInt32)(303)
const CUDA_ERROR_OPERATING_SYSTEM = (UInt32)(304)
const CUDA_ERROR_INVALID_HANDLE = (UInt32)(400)
const CUDA_ERROR_NOT_FOUND = (UInt32)(500)
const CUDA_ERROR_NOT_READY = (UInt32)(600)
const CUDA_ERROR_ILLEGAL_ADDRESS = (UInt32)(700)
const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = (UInt32)(701)
const CUDA_ERROR_LAUNCH_TIMEOUT = (UInt32)(702)
const CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = (UInt32)(703)
const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = (UInt32)(704)
const CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = (UInt32)(705)
const CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = (UInt32)(708)
const CUDA_ERROR_CONTEXT_IS_DESTROYED = (UInt32)(709)
const CUDA_ERROR_ASSERT = (UInt32)(710)
const CUDA_ERROR_TOO_MANY_PEERS = (UInt32)(711)
const CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = (UInt32)(712)
const CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = (UInt32)(713)
const CUDA_ERROR_HARDWARE_STACK_ERROR = (UInt32)(714)
const CUDA_ERROR_ILLEGAL_INSTRUCTION = (UInt32)(715)
const CUDA_ERROR_MISALIGNED_ADDRESS = (UInt32)(716)
const CUDA_ERROR_INVALID_ADDRESS_SPACE = (UInt32)(717)
const CUDA_ERROR_INVALID_PC = (UInt32)(718)
const CUDA_ERROR_LAUNCH_FAILED = (UInt32)(719)
const CUDA_ERROR_NOT_PERMITTED = (UInt32)(800)
const CUDA_ERROR_NOT_SUPPORTED = (UInt32)(801)
const CUDA_ERROR_UNKNOWN = (UInt32)(999)
# end enum CUresult

const td_sym = td_target
const td_sym = td_target

immutable CUDA_MEMCPY2D_st
    srcXInBytes::Csize_t
    srcY::Csize_t
    srcMemoryType::CUmemorytype
    srcHost::Ptr{Void}
    srcDevice::CUdeviceptr
    srcArray::CUarray
    srcPitch::Csize_t
    dstXInBytes::Csize_t
    dstY::Csize_t
    dstMemoryType::CUmemorytype
    dstHost::Ptr{Void}
    dstDevice::CUdeviceptr
    dstArray::CUarray
    dstPitch::Csize_t
    WidthInBytes::Csize_t
    Height::Csize_t
end

immutable CUDA_MEMCPY2D
    srcXInBytes::Csize_t
    srcY::Csize_t
    srcMemoryType::CUmemorytype
    srcHost::Ptr{Void}
    srcDevice::CUdeviceptr
    srcArray::CUarray
    srcPitch::Csize_t
    dstXInBytes::Csize_t
    dstY::Csize_t
    dstMemoryType::CUmemorytype
    dstHost::Ptr{Void}
    dstDevice::CUdeviceptr
    dstArray::CUarray
    dstPitch::Csize_t
    WidthInBytes::Csize_t
    Height::Csize_t
end

immutable CUDA_MEMCPY3D_st
    srcXInBytes::Csize_t
    srcY::Csize_t
    srcZ::Csize_t
    srcLOD::Csize_t
    srcMemoryType::CUmemorytype
    srcHost::Ptr{Void}
    srcDevice::CUdeviceptr
    srcArray::CUarray
    reserved0::Ptr{Void}
    srcPitch::Csize_t
    srcHeight::Csize_t
    dstXInBytes::Csize_t
    dstY::Csize_t
    dstZ::Csize_t
    dstLOD::Csize_t
    dstMemoryType::CUmemorytype
    dstHost::Ptr{Void}
    dstDevice::CUdeviceptr
    dstArray::CUarray
    reserved1::Ptr{Void}
    dstPitch::Csize_t
    dstHeight::Csize_t
    WidthInBytes::Csize_t
    Height::Csize_t
    Depth::Csize_t
end

immutable CUDA_MEMCPY3D
    srcXInBytes::Csize_t
    srcY::Csize_t
    srcZ::Csize_t
    srcLOD::Csize_t
    srcMemoryType::CUmemorytype
    srcHost::Ptr{Void}
    srcDevice::CUdeviceptr
    srcArray::CUarray
    reserved0::Ptr{Void}
    srcPitch::Csize_t
    srcHeight::Csize_t
    dstXInBytes::Csize_t
    dstY::Csize_t
    dstZ::Csize_t
    dstLOD::Csize_t
    dstMemoryType::CUmemorytype
    dstHost::Ptr{Void}
    dstDevice::CUdeviceptr
    dstArray::CUarray
    reserved1::Ptr{Void}
    dstPitch::Csize_t
    dstHeight::Csize_t
    WidthInBytes::Csize_t
    Height::Csize_t
    Depth::Csize_t
end

immutable CUDA_MEMCPY3D_PEER_st
    srcXInBytes::Csize_t
    srcY::Csize_t
    srcZ::Csize_t
    srcLOD::Csize_t
    srcMemoryType::CUmemorytype
    srcHost::Ptr{Void}
    srcDevice::CUdeviceptr
    srcArray::CUarray
    srcContext::CUcontext
    srcPitch::Csize_t
    srcHeight::Csize_t
    dstXInBytes::Csize_t
    dstY::Csize_t
    dstZ::Csize_t
    dstLOD::Csize_t
    dstMemoryType::CUmemorytype
    dstHost::Ptr{Void}
    dstDevice::CUdeviceptr
    dstArray::CUarray
    dstContext::CUcontext
    dstPitch::Csize_t
    dstHeight::Csize_t
    WidthInBytes::Csize_t
    Height::Csize_t
    Depth::Csize_t
end

immutable CUDA_MEMCPY3D_PEER
    srcXInBytes::Csize_t
    srcY::Csize_t
    srcZ::Csize_t
    srcLOD::Csize_t
    srcMemoryType::CUmemorytype
    srcHost::Ptr{Void}
    srcDevice::CUdeviceptr
    srcArray::CUarray
    srcContext::CUcontext
    srcPitch::Csize_t
    srcHeight::Csize_t
    dstXInBytes::Csize_t
    dstY::Csize_t
    dstZ::Csize_t
    dstLOD::Csize_t
    dstMemoryType::CUmemorytype
    dstHost::Ptr{Void}
    dstDevice::CUdeviceptr
    dstArray::CUarray
    dstContext::CUcontext
    dstPitch::Csize_t
    dstHeight::Csize_t
    WidthInBytes::Csize_t
    Height::Csize_t
    Depth::Csize_t
end

immutable CUDA_ARRAY_DESCRIPTOR_st
    Width::Csize_t
    Height::Csize_t
    Format::CUarray_format
    NumChannels::UInt32
end

immutable CUDA_ARRAY_DESCRIPTOR
    Width::Csize_t
    Height::Csize_t
    Format::CUarray_format
    NumChannels::UInt32
end

immutable CUDA_ARRAY3D_DESCRIPTOR_st
    Width::Csize_t
    Height::Csize_t
    Depth::Csize_t
    Format::CUarray_format
    NumChannels::UInt32
    Flags::UInt32
end

immutable CUDA_ARRAY3D_DESCRIPTOR
    Width::Csize_t
    Height::Csize_t
    Depth::Csize_t
    Format::CUarray_format
    NumChannels::UInt32
    Flags::UInt32
end

immutable CUDA_RESOURCE_DESC_st
    resType::CUresourcetype
    res::Void
    flags::UInt32
end

immutable CUDA_RESOURCE_DESC
    resType::CUresourcetype
    res::Void
    flags::UInt32
end

immutable Array_3_CUaddress_mode
    d1::CUaddress_mode
    d2::CUaddress_mode
    d3::CUaddress_mode
end

zero(::Type{Array_3_CUaddress_mode}) = begin  # /home/cl/shindo/lemon/.julia/v0.4/Clang/src/wrap_c.jl, line 266:
        Array_3_CUaddress_mode(fill(zero(CUaddress_mode),3)...)
    end

immutable Array_16_Cint
    d1::Cint
    d2::Cint
    d3::Cint
    d4::Cint
    d5::Cint
    d6::Cint
    d7::Cint
    d8::Cint
    d9::Cint
    d10::Cint
    d11::Cint
    d12::Cint
    d13::Cint
    d14::Cint
    d15::Cint
    d16::Cint
end

zero(::Type{Array_16_Cint}) = begin  # /home/cl/shindo/lemon/.julia/v0.4/Clang/src/wrap_c.jl, line 266:
        Array_16_Cint(fill(zero(Cint),16)...)
    end

immutable CUDA_TEXTURE_DESC_st
    addressMode::Array_3_CUaddress_mode
    filterMode::CUfilter_mode
    flags::UInt32
    maxAnisotropy::UInt32
    mipmapFilterMode::CUfilter_mode
    mipmapLevelBias::Cfloat
    minMipmapLevelClamp::Cfloat
    maxMipmapLevelClamp::Cfloat
    reserved::Array_16_Cint
end

immutable CUDA_TEXTURE_DESC
    addressMode::Array_3_CUaddress_mode
    filterMode::CUfilter_mode
    flags::UInt32
    maxAnisotropy::UInt32
    mipmapFilterMode::CUfilter_mode
    mipmapLevelBias::Cfloat
    minMipmapLevelClamp::Cfloat
    maxMipmapLevelClamp::Cfloat
    reserved::Array_16_Cint
end

# begin enum CUresourceViewFormat_enum
const CUresourceViewFormat_enum = UInt32
const CU_RES_VIEW_FORMAT_NONE = (UInt32)(0)
const CU_RES_VIEW_FORMAT_UINT_1X8 = (UInt32)(1)
const CU_RES_VIEW_FORMAT_UINT_2X8 = (UInt32)(2)
const CU_RES_VIEW_FORMAT_UINT_4X8 = (UInt32)(3)
const CU_RES_VIEW_FORMAT_SINT_1X8 = (UInt32)(4)
const CU_RES_VIEW_FORMAT_SINT_2X8 = (UInt32)(5)
const CU_RES_VIEW_FORMAT_SINT_4X8 = (UInt32)(6)
const CU_RES_VIEW_FORMAT_UINT_1X16 = (UInt32)(7)
const CU_RES_VIEW_FORMAT_UINT_2X16 = (UInt32)(8)
const CU_RES_VIEW_FORMAT_UINT_4X16 = (UInt32)(9)
const CU_RES_VIEW_FORMAT_SINT_1X16 = (UInt32)(10)
const CU_RES_VIEW_FORMAT_SINT_2X16 = (UInt32)(11)
const CU_RES_VIEW_FORMAT_SINT_4X16 = (UInt32)(12)
const CU_RES_VIEW_FORMAT_UINT_1X32 = (UInt32)(13)
const CU_RES_VIEW_FORMAT_UINT_2X32 = (UInt32)(14)
const CU_RES_VIEW_FORMAT_UINT_4X32 = (UInt32)(15)
const CU_RES_VIEW_FORMAT_SINT_1X32 = (UInt32)(16)
const CU_RES_VIEW_FORMAT_SINT_2X32 = (UInt32)(17)
const CU_RES_VIEW_FORMAT_SINT_4X32 = (UInt32)(18)
const CU_RES_VIEW_FORMAT_FLOAT_1X16 = (UInt32)(19)
const CU_RES_VIEW_FORMAT_FLOAT_2X16 = (UInt32)(20)
const CU_RES_VIEW_FORMAT_FLOAT_4X16 = (UInt32)(21)
const CU_RES_VIEW_FORMAT_FLOAT_1X32 = (UInt32)(22)
const CU_RES_VIEW_FORMAT_FLOAT_2X32 = (UInt32)(23)
const CU_RES_VIEW_FORMAT_FLOAT_4X32 = (UInt32)(24)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = (UInt32)(25)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = (UInt32)(26)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = (UInt32)(27)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = (UInt32)(28)
const CU_RES_VIEW_FORMAT_SIGNED_BC4 = (UInt32)(29)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = (UInt32)(30)
const CU_RES_VIEW_FORMAT_SIGNED_BC5 = (UInt32)(31)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = (UInt32)(32)
const CU_RES_VIEW_FORMAT_SIGNED_BC6H = (UInt32)(33)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = (UInt32)(34)
# end enum CUresourceViewFormat_enum

# begin enum CUresourceViewFormat
const CUresourceViewFormat = UInt32
const CU_RES_VIEW_FORMAT_NONE = (UInt32)(0)
const CU_RES_VIEW_FORMAT_UINT_1X8 = (UInt32)(1)
const CU_RES_VIEW_FORMAT_UINT_2X8 = (UInt32)(2)
const CU_RES_VIEW_FORMAT_UINT_4X8 = (UInt32)(3)
const CU_RES_VIEW_FORMAT_SINT_1X8 = (UInt32)(4)
const CU_RES_VIEW_FORMAT_SINT_2X8 = (UInt32)(5)
const CU_RES_VIEW_FORMAT_SINT_4X8 = (UInt32)(6)
const CU_RES_VIEW_FORMAT_UINT_1X16 = (UInt32)(7)
const CU_RES_VIEW_FORMAT_UINT_2X16 = (UInt32)(8)
const CU_RES_VIEW_FORMAT_UINT_4X16 = (UInt32)(9)
const CU_RES_VIEW_FORMAT_SINT_1X16 = (UInt32)(10)
const CU_RES_VIEW_FORMAT_SINT_2X16 = (UInt32)(11)
const CU_RES_VIEW_FORMAT_SINT_4X16 = (UInt32)(12)
const CU_RES_VIEW_FORMAT_UINT_1X32 = (UInt32)(13)
const CU_RES_VIEW_FORMAT_UINT_2X32 = (UInt32)(14)
const CU_RES_VIEW_FORMAT_UINT_4X32 = (UInt32)(15)
const CU_RES_VIEW_FORMAT_SINT_1X32 = (UInt32)(16)
const CU_RES_VIEW_FORMAT_SINT_2X32 = (UInt32)(17)
const CU_RES_VIEW_FORMAT_SINT_4X32 = (UInt32)(18)
const CU_RES_VIEW_FORMAT_FLOAT_1X16 = (UInt32)(19)
const CU_RES_VIEW_FORMAT_FLOAT_2X16 = (UInt32)(20)
const CU_RES_VIEW_FORMAT_FLOAT_4X16 = (UInt32)(21)
const CU_RES_VIEW_FORMAT_FLOAT_1X32 = (UInt32)(22)
const CU_RES_VIEW_FORMAT_FLOAT_2X32 = (UInt32)(23)
const CU_RES_VIEW_FORMAT_FLOAT_4X32 = (UInt32)(24)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = (UInt32)(25)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = (UInt32)(26)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = (UInt32)(27)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = (UInt32)(28)
const CU_RES_VIEW_FORMAT_SIGNED_BC4 = (UInt32)(29)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = (UInt32)(30)
const CU_RES_VIEW_FORMAT_SIGNED_BC5 = (UInt32)(31)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = (UInt32)(32)
const CU_RES_VIEW_FORMAT_SIGNED_BC6H = (UInt32)(33)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = (UInt32)(34)
# end enum CUresourceViewFormat

immutable Array_16_UInt32
    d1::UInt32
    d2::UInt32
    d3::UInt32
    d4::UInt32
    d5::UInt32
    d6::UInt32
    d7::UInt32
    d8::UInt32
    d9::UInt32
    d10::UInt32
    d11::UInt32
    d12::UInt32
    d13::UInt32
    d14::UInt32
    d15::UInt32
    d16::UInt32
end

zero(::Type{Array_16_UInt32}) = begin  # /home/cl/shindo/lemon/.julia/v0.4/Clang/src/wrap_c.jl, line 266:
        Array_16_UInt32(fill(zero(UInt32),16)...)
    end

immutable CUDA_RESOURCE_VIEW_DESC_st
    format::CUresourceViewFormat
    width::Csize_t
    height::Csize_t
    depth::Csize_t
    firstMipmapLevel::UInt32
    lastMipmapLevel::UInt32
    firstLayer::UInt32
    lastLayer::UInt32
    reserved::Array_16_UInt32
end

immutable CUDA_RESOURCE_VIEW_DESC
    format::CUresourceViewFormat
    width::Csize_t
    height::Csize_t
    depth::Csize_t
    firstMipmapLevel::UInt32
    lastMipmapLevel::UInt32
    firstLayer::UInt32
    lastLayer::UInt32
    reserved::Array_16_UInt32
end

immutable CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
    p2pToken::Culonglong
    vaSpaceToken::UInt32
end

immutable CUDA_POINTER_ATTRIBUTE_P2P_TOKENS
    p2pToken::Culonglong
    vaSpaceToken::UInt32
end
