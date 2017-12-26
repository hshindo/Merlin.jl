const define = Dict{Symbol,Symbol}()

if API_VERSION >= 3020
    define[:cuDeviceTotalMem]           = :cuDeviceTotalMem_v2
    define[:cuCtxCreate]                = :cuCtxCreate_v2
    define[:cuModuleGetGlobal]          = :cuModuleGetGlobal_v2
    define[:cuMemGetInfo]               = :cuMemGetInfo_v2
    define[:cuMemAlloc]                 = :cuMemAlloc_v2
    define[:cuMemAllocPitch]            = :cuMemAllocPitch_v2
    define[:cuMemFree]                  = :cuMemFree_v2
    define[:cuMemGetAddressRange]       = :cuMemGetAddressRange_v2
    define[:cuMemAllocHost]             = :cuMemAllocHost_v2
    define[:cuMemHostGetDevicePointer]  = :cuMemHostGetDevicePointer_v2
    define[:cuMemcpyHtoD]               = :cuMemcpyHtoD_v2
    define[:cuMemcpyDtoH]               = :cuMemcpyDtoH_v2
    define[:cuMemcpyDtoD]               = :cuMemcpyDtoD_v2
    define[:cuMemcpyDtoA]               = :cuMemcpyDtoA_v2
    define[:cuMemcpyAtoD]               = :cuMemcpyAtoD_v2
    define[:cuMemcpyHtoA]               = :cuMemcpyHtoA_v2
    define[:cuMemcpyAtoH]               = :cuMemcpyAtoH_v2
    define[:cuMemcpyAtoA]               = :cuMemcpyAtoA_v2
    define[:cuMemcpyHtoAAsync]          = :cuMemcpyHtoAAsync_v2
    define[:cuMemcpyAtoHAsync]          = :cuMemcpyAtoHAsync_v2
    define[:cuMemcpy2D]                 = :cuMemcpy2D_v2
    define[:cuMemcpy2DUnaligned]        = :cuMemcpy2DUnaligned_v2
    define[:cuMemcpy3D]                 = :cuMemcpy3D_v2
    define[:cuMemcpyHtoDAsync]          = :cuMemcpyHtoDAsync_v2
    define[:cuMemcpyDtoHAsync]          = :cuMemcpyDtoHAsync_v2
    define[:cuMemcpyDtoDAsync]          = :cuMemcpyDtoDAsync_v2
    define[:cuMemcpy2DAsync]            = :cuMemcpy2DAsync_v2
    define[:cuMemcpy3DAsync]            = :cuMemcpy3DAsync_v2
    define[:cuMemsetD8]                 = :cuMemsetD8_v2
    define[:cuMemsetD16]                = :cuMemsetD16_v2
    define[:cuMemsetD32]                = :cuMemsetD32_v2
    define[:cuMemsetD2D8]               = :cuMemsetD2D8_v2
    define[:cuMemsetD2D16]              = :cuMemsetD2D16_v2
    define[:cuMemsetD2D32]              = :cuMemsetD2D32_v2
    define[:cuArrayCreate]              = :cuArrayCreate_v2
    define[:cuArrayGetDescriptor]       = :cuArrayGetDescriptor_v2
    define[:cuArray3DCreate]            = :cuArray3DCreate_v2
    define[:cuArray3DGetDescriptor]     = :cuArray3DGetDescriptor_v2
    define[:cuTexRefSetAddress]         = :cuTexRefSetAddress_v2
    define[:cuTexRefGetAddress]         = :cuTexRefGetAddress_v2
    define[:cuGraphicsResourceGetMappedPointer] = :cuGraphicsResourceGetMappedPointer_v2
end
if API_VERSION >= 4000
    define[:cuCtxDestroy]               = :cuCtxDestroy_v2
    define[:cuCtxPopCurrent]            = :cuCtxPopCurrent_v2
    define[:cuCtxPushCurrent]           = :cuCtxPushCurrent_v2
    define[:cuStreamDestroy]            = :cuStreamDestroy_v2
    define[:cuEventDestroy]             = :cuEventDestroy_v2
end
if API_VERSION >= 4010
    define[:cuTexRefSetAddress2D]       = :cuTexRefSetAddress2D_v3
end
if API_VERSION >= 6050
    define[:cuLinkCreate]              = :cuLinkCreate_v2
    define[:cuLinkAddData]             = :cuLinkAddData_v2
    define[:cuLinkAddFile]             = :cuLinkAddFile_v2
end
if API_VERSION >= 6050
    define[:cuMemHostRegister]         = :cuMemHostRegister_v2
    define[:cuGraphicsResourceSetMapFlags] = :cuGraphicsResourceSetMapFlags_v2
end
if 3020 <= API_VERSION < 4010
    define[:cuTexRefSetAddress2D]      = :cuTexRefSetAddress2D_v2
end
