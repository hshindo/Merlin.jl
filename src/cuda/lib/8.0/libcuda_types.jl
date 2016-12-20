# Automatically generated using Clang.jl wrap_c, version 0.0.0

#using Compat

const cuDeviceTotalMem = cuDeviceTotalMem_v2
const cuCtxCreate = cuCtxCreate_v2
const cuModuleGetGlobal = cuModuleGetGlobal_v2
const cuMemGetInfo = cuMemGetInfo_v2
const cuMemAlloc = cuMemAlloc_v2
const cuMemAllocPitch = cuMemAllocPitch_v2
const cuMemFree = cuMemFree_v2
const cuMemGetAddressRange = cuMemGetAddressRange_v2
const cuMemAllocHost = cuMemAllocHost_v2
const cuMemHostGetDevicePointer = cuMemHostGetDevicePointer_v2

const cuMemcpyHtoD = cuMemcpyHtoD_v2
const cuMemcpyDtoH = cuMemcpyDtoH_v2
const cuMemcpyDtoD = cuMemcpyDtoD_v2
const cuMemcpyDtoA = cuMemcpyDtoA_v2
const cuMemcpyAtoD = cuMemcpyAtoD_v2
const cuMemcpyHtoA = cuMemcpyHtoA_v2
const cuMemcpyAtoH = cuMemcpyAtoH_v2
const cuMemcpyAtoA = cuMemcpyAtoA_v2
const cuMemcpyHtoAAsync = cuMemcpyHtoAAsync_v2
const cuMemcpyAtoHAsync = cuMemcpyAtoHAsync_v2
const cuMemcpy2D = cuMemcpy2D_v2
const cuMemcpy2DUnaligned = cuMemcpy2DUnaligned_v2
const cuMemcpy3D= cuMemcpy3D_v2
const cuMemcpyHtoDAsync = cuMemcpyHtoDAsync_v2
const cuMemcpyDtoHAsync = cuMemcpyDtoHAsync_v2
const cuMemcpyDtoDAsync = cuMemcpyDtoDAsync_v2
const cuMemcpy2DAsync = cuMemcpy2DAsync_v2
const cuMemcpy3DAsync = cuMemcpy3DAsync_v2
const cuMemsetD8 = cuMemsetD8_v2
const cuMemsetD16 = cuMemsetD16_v2
const cuMemsetD32 = cuMemsetD32_v2
const cuMemsetD2D8 = cuMemsetD2D8_v2
const cuMemsetD2D16 = cuMemsetD2D16_v2
const cuMemsetD2D32 = cuMemsetD2D32_v2

const cuArrayCreate = cuArrayCreate_v2
const cuArrayGetDescriptor = cuArrayGetDescriptor_v2
const cuArray3DCreate = cuArray3DCreate_v2
const cuArray3DGetDescriptor = cuArray3DGetDescriptor_v2
const cuTexRefSetAddress = cuTexRefSetAddress_v2
const cuTexRefGetAddress = cuTexRefGetAddress_v2
const cuGraphicsResourceGetMappedPointer = cuGraphicsResourceGetMappedPointer_v2
const cuCtxDestroy = cuCtxDestroy_v2
const cuCtxPopCurrent = cuCtxPopCurrent_v2
const cuCtxPushCurrent = cuCtxPushCurrent_v2
const cuStreamDestroy = cuStreamDestroy_v2
const cuEventDestroy = cuEventDestroy_v2
const cuTexRefSetAddress2D = cuTexRefSetAddress2D_v3
const cuLinkCreate = cuLinkCreate_v2
const cuLinkAddData = cuLinkAddData_v2
const cuLinkAddFile = cuLinkAddFile_v2
const cuMemHostRegister = cuMemHostRegister_v2
const cuGraphicsResourceSetMapFlags = cuGraphicsResourceSetMapFlags_v2
const CUDA_VERSION = 8000
const CU_IPC_HANDLE_SIZE = 64

# Skipping MacroDefinition: CU_STREAM_LEGACY ( ( CUstream ) 0x1 )
# Skipping MacroDefinition: CU_STREAM_PER_THREAD ( ( CUstream ) 0x2 )

const CU_MEMHOSTALLOC_PORTABLE = 0x01
const CU_MEMHOSTALLOC_DEVICEMAP = 0x02
const CU_MEMHOSTALLOC_WRITECOMBINED = 0x04
const CU_MEMHOSTREGISTER_PORTABLE = 0x01
const CU_MEMHOSTREGISTER_DEVICEMAP = 0x02
const CU_MEMHOSTREGISTER_IOMEMORY = 0x04
const CUDA_ARRAY3D_LAYERED = 0x01
const CUDA_ARRAY3D_2DARRAY = 0x01
const CUDA_ARRAY3D_SURFACE_LDST = 0x02
const CUDA_ARRAY3D_CUBEMAP = 0x04
const CUDA_ARRAY3D_TEXTURE_GATHER = 0x08
const CUDA_ARRAY3D_DEPTH_TEXTURE = 0x10
const CU_TRSA_OVERRIDE_FORMAT = 0x01
const CU_TRSF_READ_AS_INTEGER = 0x01
const CU_TRSF_NORMALIZED_COORDINATES = 0x02
const CU_TRSF_SRGB = 0x10

# Skipping MacroDefinition: CU_LAUNCH_PARAM_END ( ( void * ) 0x00 )
# Skipping MacroDefinition: CU_LAUNCH_PARAM_BUFFER_POINTER ( ( void * ) 0x01 )
# Skipping MacroDefinition: CU_LAUNCH_PARAM_BUFFER_SIZE ( ( void * ) 0x02 )

const CU_PARAM_TR_DEFAULT = -1

# Skipping MacroDefinition: CU_DEVICE_CPU ( ( CUdevice ) - 1 )
# Skipping MacroDefinition: CU_DEVICE_INVALID ( ( CUdevice ) - 2 )

typealias cuuint32_t UInt32
typealias cuuint64_t UInt64
typealias CUdeviceptr Culonglong
typealias CUdevice Cint
typealias CUctx_st Void
typealias CUcontext Ptr{CUctx_st}
typealias CUmod_st Void
typealias CUmodule Ptr{CUmod_st}
typealias CUfunc_st Void
typealias CUfunction Ptr{CUfunc_st}
typealias CUarray_st Void
typealias CUarray Ptr{CUarray_st}
typealias CUmipmappedArray_st Void
typealias CUmipmappedArray Ptr{CUmipmappedArray_st}
typealias CUtexref_st Void
typealias CUtexref Ptr{CUtexref_st}
typealias CUsurfref_st Void
typealias CUsurfref Ptr{CUsurfref_st}
typealias CUevent_st Void
typealias CUevent Ptr{CUevent_st}
typealias CUstream_st Void
typealias CUstream Ptr{CUstream_st}
typealias CUgraphicsResource_st Void
typealias CUgraphicsResource Ptr{CUgraphicsResource_st}
typealias CUtexObject Culonglong
typealias CUsurfObject Culonglong

immutable Array_16_UInt8
    d1::UInt8
    d2::UInt8
    d3::UInt8
    d4::UInt8
    d5::UInt8
    d6::UInt8
    d7::UInt8
    d8::UInt8
    d9::UInt8
    d10::UInt8
    d11::UInt8
    d12::UInt8
    d13::UInt8
    d14::UInt8
    d15::UInt8
    d16::UInt8
end

zero(::Type{Array_16_UInt8}) = begin  # /home/shindo/local-lemon/.julia/v0.4/Clang/src/wrap_c.jl, line 266:
        Array_16_UInt8(fill(zero(UInt8),16)...)
    end

immutable CUuuid_st
    bytes::Array_16_UInt8
end

immutable CUuuid
    bytes::Array_16_UInt8
end

immutable Array_64_UInt8
    d1::UInt8
    d2::UInt8
    d3::UInt8
    d4::UInt8
    d5::UInt8
    d6::UInt8
    d7::UInt8
    d8::UInt8
    d9::UInt8
    d10::UInt8
    d11::UInt8
    d12::UInt8
    d13::UInt8
    d14::UInt8
    d15::UInt8
    d16::UInt8
    d17::UInt8
    d18::UInt8
    d19::UInt8
    d20::UInt8
    d21::UInt8
    d22::UInt8
    d23::UInt8
    d24::UInt8
    d25::UInt8
    d26::UInt8
    d27::UInt8
    d28::UInt8
    d29::UInt8
    d30::UInt8
    d31::UInt8
    d32::UInt8
    d33::UInt8
    d34::UInt8
    d35::UInt8
    d36::UInt8
    d37::UInt8
    d38::UInt8
    d39::UInt8
    d40::UInt8
    d41::UInt8
    d42::UInt8
    d43::UInt8
    d44::UInt8
    d45::UInt8
    d46::UInt8
    d47::UInt8
    d48::UInt8
    d49::UInt8
    d50::UInt8
    d51::UInt8
    d52::UInt8
    d53::UInt8
    d54::UInt8
    d55::UInt8
    d56::UInt8
    d57::UInt8
    d58::UInt8
    d59::UInt8
    d60::UInt8
    d61::UInt8
    d62::UInt8
    d63::UInt8
    d64::UInt8
end

zero(::Type{Array_64_UInt8}) = begin  # /home/shindo/local-lemon/.julia/v0.4/Clang/src/wrap_c.jl, line 266:
        Array_64_UInt8(fill(zero(UInt8),64)...)
    end

immutable CUipcEventHandle_st
    reserved::Array_64_UInt8
end

immutable CUipcEventHandle
    reserved::Array_64_UInt8
end

immutable CUipcMemHandle_st
    reserved::Array_64_UInt8
end

immutable CUipcMemHandle
    reserved::Array_64_UInt8
end

# begin enum CUipcMem_flags_enum
typealias CUipcMem_flags_enum UInt32
const CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = (UInt32)(1)
# end enum CUipcMem_flags_enum

# begin enum CUipcMem_flags
typealias CUipcMem_flags UInt32
const CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = (UInt32)(1)
# end enum CUipcMem_flags

# begin enum CUmemAttach_flags_enum
typealias CUmemAttach_flags_enum UInt32
const CU_MEM_ATTACH_GLOBAL = (UInt32)(1)
const CU_MEM_ATTACH_HOST = (UInt32)(2)
const CU_MEM_ATTACH_SINGLE = (UInt32)(4)
# end enum CUmemAttach_flags_enum

# begin enum CUmemAttach_flags
typealias CUmemAttach_flags UInt32
const CU_MEM_ATTACH_GLOBAL = (UInt32)(1)
const CU_MEM_ATTACH_HOST = (UInt32)(2)
const CU_MEM_ATTACH_SINGLE = (UInt32)(4)
# end enum CUmemAttach_flags

# begin enum CUctx_flags_enum
typealias CUctx_flags_enum UInt32
const CU_CTX_SCHED_AUTO = (UInt32)(0)
const CU_CTX_SCHED_SPIN = (UInt32)(1)
const CU_CTX_SCHED_YIELD = (UInt32)(2)
const CU_CTX_SCHED_BLOCKING_SYNC = (UInt32)(4)
const CU_CTX_BLOCKING_SYNC = (UInt32)(4)
const CU_CTX_SCHED_MASK = (UInt32)(7)
const CU_CTX_MAP_HOST = (UInt32)(8)
const CU_CTX_LMEM_RESIZE_TO_MAX = (UInt32)(16)
const CU_CTX_FLAGS_MASK = (UInt32)(31)
# end enum CUctx_flags_enum

# begin enum CUctx_flags
typealias CUctx_flags UInt32
const CU_CTX_SCHED_AUTO = (UInt32)(0)
const CU_CTX_SCHED_SPIN = (UInt32)(1)
const CU_CTX_SCHED_YIELD = (UInt32)(2)
const CU_CTX_SCHED_BLOCKING_SYNC = (UInt32)(4)
const CU_CTX_BLOCKING_SYNC = (UInt32)(4)
const CU_CTX_SCHED_MASK = (UInt32)(7)
const CU_CTX_MAP_HOST = (UInt32)(8)
const CU_CTX_LMEM_RESIZE_TO_MAX = (UInt32)(16)
const CU_CTX_FLAGS_MASK = (UInt32)(31)
# end enum CUctx_flags

# begin enum CUstream_flags_enum
typealias CUstream_flags_enum UInt32
const CU_STREAM_DEFAULT = (UInt32)(0)
const CU_STREAM_NON_BLOCKING = (UInt32)(1)
# end enum CUstream_flags_enum

# begin enum CUstream_flags
typealias CUstream_flags UInt32
const CU_STREAM_DEFAULT = (UInt32)(0)
const CU_STREAM_NON_BLOCKING = (UInt32)(1)
# end enum CUstream_flags

# begin enum CUevent_flags_enum
typealias CUevent_flags_enum UInt32
const CU_EVENT_DEFAULT = (UInt32)(0)
const CU_EVENT_BLOCKING_SYNC = (UInt32)(1)
const CU_EVENT_DISABLE_TIMING = (UInt32)(2)
const CU_EVENT_INTERPROCESS = (UInt32)(4)
# end enum CUevent_flags_enum

# begin enum CUevent_flags
typealias CUevent_flags UInt32
const CU_EVENT_DEFAULT = (UInt32)(0)
const CU_EVENT_BLOCKING_SYNC = (UInt32)(1)
const CU_EVENT_DISABLE_TIMING = (UInt32)(2)
const CU_EVENT_INTERPROCESS = (UInt32)(4)
# end enum CUevent_flags

# begin enum CUstreamWaitValue_flags_enum
typealias CUstreamWaitValue_flags_enum UInt32
const CU_STREAM_WAIT_VALUE_GEQ = (UInt32)(0)
const CU_STREAM_WAIT_VALUE_EQ = (UInt32)(1)
const CU_STREAM_WAIT_VALUE_AND = (UInt32)(2)
const CU_STREAM_WAIT_VALUE_FLUSH = (UInt32)(1073741824)
# end enum CUstreamWaitValue_flags_enum

# begin enum CUstreamWaitValue_flags
typealias CUstreamWaitValue_flags UInt32
const CU_STREAM_WAIT_VALUE_GEQ = (UInt32)(0)
const CU_STREAM_WAIT_VALUE_EQ = (UInt32)(1)
const CU_STREAM_WAIT_VALUE_AND = (UInt32)(2)
const CU_STREAM_WAIT_VALUE_FLUSH = (UInt32)(1073741824)
# end enum CUstreamWaitValue_flags

# begin enum CUstreamWriteValue_flags_enum
typealias CUstreamWriteValue_flags_enum UInt32
const CU_STREAM_WRITE_VALUE_DEFAULT = (UInt32)(0)
const CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = (UInt32)(1)
# end enum CUstreamWriteValue_flags_enum

# begin enum CUstreamWriteValue_flags
typealias CUstreamWriteValue_flags UInt32
const CU_STREAM_WRITE_VALUE_DEFAULT = (UInt32)(0)
const CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = (UInt32)(1)
# end enum CUstreamWriteValue_flags

# begin enum CUstreamBatchMemOpType_enum
typealias CUstreamBatchMemOpType_enum UInt32
const CU_STREAM_MEM_OP_WAIT_VALUE_32 = (UInt32)(1)
const CU_STREAM_MEM_OP_WRITE_VALUE_32 = (UInt32)(2)
const CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = (UInt32)(3)
# end enum CUstreamBatchMemOpType_enum

# begin enum CUstreamBatchMemOpType
typealias CUstreamBatchMemOpType UInt32
const CU_STREAM_MEM_OP_WAIT_VALUE_32 = (UInt32)(1)
const CU_STREAM_MEM_OP_WRITE_VALUE_32 = (UInt32)(2)
const CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = (UInt32)(3)
# end enum CUstreamBatchMemOpType

immutable Array_6_cuuint64_t
    d1::cuuint64_t
    d2::cuuint64_t
    d3::cuuint64_t
    d4::cuuint64_t
    d5::cuuint64_t
    d6::cuuint64_t
end

zero(::Type{Array_6_cuuint64_t}) = begin  # /home/shindo/local-lemon/.julia/v0.4/Clang/src/wrap_c.jl, line 266:
        Array_6_cuuint64_t(fill(zero(cuuint64_t),6)...)
    end

immutable CUstreamBatchMemOpParams_union
    _CUstreamBatchMemOpParams_union::Array_6_cuuint64_t
end

immutable CUstreamBatchMemOpParams
    _CUstreamBatchMemOpParams::Array_6_cuuint64_t
end

# begin enum CUoccupancy_flags_enum
typealias CUoccupancy_flags_enum UInt32
const CU_OCCUPANCY_DEFAULT = (UInt32)(0)
const CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = (UInt32)(1)
# end enum CUoccupancy_flags_enum

# begin enum CUoccupancy_flags
typealias CUoccupancy_flags UInt32
const CU_OCCUPANCY_DEFAULT = (UInt32)(0)
const CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = (UInt32)(1)
# end enum CUoccupancy_flags

# begin enum CUarray_format_enum
typealias CUarray_format_enum UInt32
const CU_AD_FORMAT_UNSIGNED_INT8 = (UInt32)(1)
const CU_AD_FORMAT_UNSIGNED_INT16 = (UInt32)(2)
const CU_AD_FORMAT_UNSIGNED_INT32 = (UInt32)(3)
const CU_AD_FORMAT_SIGNED_INT8 = (UInt32)(8)
const CU_AD_FORMAT_SIGNED_INT16 = (UInt32)(9)
const CU_AD_FORMAT_SIGNED_INT32 = (UInt32)(10)
const CU_AD_FORMAT_HALF = (UInt32)(16)
const CU_AD_FORMAT_FLOAT = (UInt32)(32)
# end enum CUarray_format_enum

# begin enum CUarray_format
typealias CUarray_format UInt32
const CU_AD_FORMAT_UNSIGNED_INT8 = (UInt32)(1)
const CU_AD_FORMAT_UNSIGNED_INT16 = (UInt32)(2)
const CU_AD_FORMAT_UNSIGNED_INT32 = (UInt32)(3)
const CU_AD_FORMAT_SIGNED_INT8 = (UInt32)(8)
const CU_AD_FORMAT_SIGNED_INT16 = (UInt32)(9)
const CU_AD_FORMAT_SIGNED_INT32 = (UInt32)(10)
const CU_AD_FORMAT_HALF = (UInt32)(16)
const CU_AD_FORMAT_FLOAT = (UInt32)(32)
# end enum CUarray_format

# begin enum CUaddress_mode_enum
typealias CUaddress_mode_enum UInt32
const CU_TR_ADDRESS_MODE_WRAP = (UInt32)(0)
const CU_TR_ADDRESS_MODE_CLAMP = (UInt32)(1)
const CU_TR_ADDRESS_MODE_MIRROR = (UInt32)(2)
const CU_TR_ADDRESS_MODE_BORDER = (UInt32)(3)
# end enum CUaddress_mode_enum

# begin enum CUaddress_mode
typealias CUaddress_mode UInt32
const CU_TR_ADDRESS_MODE_WRAP = (UInt32)(0)
const CU_TR_ADDRESS_MODE_CLAMP = (UInt32)(1)
const CU_TR_ADDRESS_MODE_MIRROR = (UInt32)(2)
const CU_TR_ADDRESS_MODE_BORDER = (UInt32)(3)
# end enum CUaddress_mode

# begin enum CUfilter_mode_enum
typealias CUfilter_mode_enum UInt32
const CU_TR_FILTER_MODE_POINT = (UInt32)(0)
const CU_TR_FILTER_MODE_LINEAR = (UInt32)(1)
# end enum CUfilter_mode_enum

# begin enum CUfilter_mode
typealias CUfilter_mode UInt32
const CU_TR_FILTER_MODE_POINT = (UInt32)(0)
const CU_TR_FILTER_MODE_LINEAR = (UInt32)(1)
# end enum CUfilter_mode

# begin enum CUdevice_attribute_enum
typealias CUdevice_attribute_enum UInt32
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = (UInt32)(1)
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = (UInt32)(2)
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = (UInt32)(3)
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = (UInt32)(4)
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = (UInt32)(5)
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = (UInt32)(6)
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = (UInt32)(7)
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = (UInt32)(8)
const CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = (UInt32)(8)
const CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = (UInt32)(9)
const CU_DEVICE_ATTRIBUTE_WARP_SIZE = (UInt32)(10)
const CU_DEVICE_ATTRIBUTE_MAX_PITCH = (UInt32)(11)
const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = (UInt32)(12)
const CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = (UInt32)(12)
const CU_DEVICE_ATTRIBUTE_CLOCK_RATE = (UInt32)(13)
const CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = (UInt32)(14)
const CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = (UInt32)(15)
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = (UInt32)(16)
const CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = (UInt32)(17)
const CU_DEVICE_ATTRIBUTE_INTEGRATED = (UInt32)(18)
const CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = (UInt32)(19)
const CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = (UInt32)(20)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = (UInt32)(21)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = (UInt32)(22)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = (UInt32)(23)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = (UInt32)(24)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = (UInt32)(25)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = (UInt32)(26)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = (UInt32)(27)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = (UInt32)(28)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = (UInt32)(29)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = (UInt32)(27)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = (UInt32)(28)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = (UInt32)(29)
const CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = (UInt32)(30)
const CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = (UInt32)(31)
const CU_DEVICE_ATTRIBUTE_ECC_ENABLED = (UInt32)(32)
const CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = (UInt32)(33)
const CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = (UInt32)(34)
const CU_DEVICE_ATTRIBUTE_TCC_DRIVER = (UInt32)(35)
const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = (UInt32)(36)
const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = (UInt32)(37)
const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = (UInt32)(38)
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = (UInt32)(39)
const CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = (UInt32)(40)
const CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = (UInt32)(41)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = (UInt32)(42)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = (UInt32)(43)
const CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = (UInt32)(44)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = (UInt32)(45)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = (UInt32)(46)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = (UInt32)(47)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = (UInt32)(48)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = (UInt32)(49)
const CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = (UInt32)(50)
const CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = (UInt32)(51)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = (UInt32)(52)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = (UInt32)(53)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = (UInt32)(54)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = (UInt32)(55)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = (UInt32)(56)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = (UInt32)(57)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = (UInt32)(58)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = (UInt32)(59)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = (UInt32)(60)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = (UInt32)(61)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = (UInt32)(62)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = (UInt32)(63)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = (UInt32)(64)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = (UInt32)(65)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = (UInt32)(66)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = (UInt32)(67)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = (UInt32)(68)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = (UInt32)(69)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = (UInt32)(70)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = (UInt32)(71)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = (UInt32)(72)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = (UInt32)(73)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = (UInt32)(74)
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = (UInt32)(75)
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = (UInt32)(76)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = (UInt32)(77)
const CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = (UInt32)(78)
const CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = (UInt32)(79)
const CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = (UInt32)(80)
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = (UInt32)(81)
const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = (UInt32)(82)
const CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = (UInt32)(83)
const CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = (UInt32)(84)
const CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = (UInt32)(85)
const CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = (UInt32)(86)
const CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = (UInt32)(87)
const CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = (UInt32)(88)
const CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = (UInt32)(89)
const CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = (UInt32)(90)
const CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = (UInt32)(91)
const CU_DEVICE_ATTRIBUTE_MAX = (UInt32)(92)
# end enum CUdevice_attribute_enum

# begin enum CUdevice_attribute
typealias CUdevice_attribute UInt32
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = (UInt32)(1)
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = (UInt32)(2)
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = (UInt32)(3)
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = (UInt32)(4)
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = (UInt32)(5)
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = (UInt32)(6)
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = (UInt32)(7)
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = (UInt32)(8)
const CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = (UInt32)(8)
const CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = (UInt32)(9)
const CU_DEVICE_ATTRIBUTE_WARP_SIZE = (UInt32)(10)
const CU_DEVICE_ATTRIBUTE_MAX_PITCH = (UInt32)(11)
const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = (UInt32)(12)
const CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = (UInt32)(12)
const CU_DEVICE_ATTRIBUTE_CLOCK_RATE = (UInt32)(13)
const CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = (UInt32)(14)
const CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = (UInt32)(15)
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = (UInt32)(16)
const CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = (UInt32)(17)
const CU_DEVICE_ATTRIBUTE_INTEGRATED = (UInt32)(18)
const CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = (UInt32)(19)
const CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = (UInt32)(20)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = (UInt32)(21)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = (UInt32)(22)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = (UInt32)(23)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = (UInt32)(24)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = (UInt32)(25)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = (UInt32)(26)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = (UInt32)(27)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = (UInt32)(28)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = (UInt32)(29)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = (UInt32)(27)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = (UInt32)(28)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = (UInt32)(29)
const CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = (UInt32)(30)
const CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = (UInt32)(31)
const CU_DEVICE_ATTRIBUTE_ECC_ENABLED = (UInt32)(32)
const CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = (UInt32)(33)
const CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = (UInt32)(34)
const CU_DEVICE_ATTRIBUTE_TCC_DRIVER = (UInt32)(35)
const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = (UInt32)(36)
const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = (UInt32)(37)
const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = (UInt32)(38)
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = (UInt32)(39)
const CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = (UInt32)(40)
const CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = (UInt32)(41)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = (UInt32)(42)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = (UInt32)(43)
const CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = (UInt32)(44)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = (UInt32)(45)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = (UInt32)(46)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = (UInt32)(47)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = (UInt32)(48)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = (UInt32)(49)
const CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = (UInt32)(50)
const CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = (UInt32)(51)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = (UInt32)(52)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = (UInt32)(53)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = (UInt32)(54)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = (UInt32)(55)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = (UInt32)(56)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = (UInt32)(57)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = (UInt32)(58)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = (UInt32)(59)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = (UInt32)(60)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = (UInt32)(61)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = (UInt32)(62)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = (UInt32)(63)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = (UInt32)(64)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = (UInt32)(65)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = (UInt32)(66)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = (UInt32)(67)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = (UInt32)(68)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = (UInt32)(69)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = (UInt32)(70)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = (UInt32)(71)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = (UInt32)(72)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = (UInt32)(73)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = (UInt32)(74)
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = (UInt32)(75)
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = (UInt32)(76)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = (UInt32)(77)
const CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = (UInt32)(78)
const CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = (UInt32)(79)
const CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = (UInt32)(80)
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = (UInt32)(81)
const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = (UInt32)(82)
const CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = (UInt32)(83)
const CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = (UInt32)(84)
const CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = (UInt32)(85)
const CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = (UInt32)(86)
const CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = (UInt32)(87)
const CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = (UInt32)(88)
const CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = (UInt32)(89)
const CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = (UInt32)(90)
const CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = (UInt32)(91)
const CU_DEVICE_ATTRIBUTE_MAX = (UInt32)(92)
# end enum CUdevice_attribute

immutable Array_3_Cint
    d1::Cint
    d2::Cint
    d3::Cint
end

zero(::Type{Array_3_Cint}) = begin  # /home/shindo/local-lemon/.julia/v0.4/Clang/src/wrap_c.jl, line 266:
        Array_3_Cint(fill(zero(Cint),3)...)
    end

immutable CUdevprop_st
    maxThreadsPerBlock::Cint
    maxThreadsDim::Array_3_Cint
    maxGridSize::Array_3_Cint
    sharedMemPerBlock::Cint
    totalConstantMemory::Cint
    SIMDWidth::Cint
    memPitch::Cint
    regsPerBlock::Cint
    clockRate::Cint
    textureAlign::Cint
end

immutable CUdevprop
    maxThreadsPerBlock::Cint
    maxThreadsDim::Array_3_Cint
    maxGridSize::Array_3_Cint
    sharedMemPerBlock::Cint
    totalConstantMemory::Cint
    SIMDWidth::Cint
    memPitch::Cint
    regsPerBlock::Cint
    clockRate::Cint
    textureAlign::Cint
end

# begin enum CUpointer_attribute_enum
typealias CUpointer_attribute_enum UInt32
const CU_POINTER_ATTRIBUTE_CONTEXT = (UInt32)(1)
const CU_POINTER_ATTRIBUTE_MEMORY_TYPE = (UInt32)(2)
const CU_POINTER_ATTRIBUTE_DEVICE_POINTER = (UInt32)(3)
const CU_POINTER_ATTRIBUTE_HOST_POINTER = (UInt32)(4)
const CU_POINTER_ATTRIBUTE_P2P_TOKENS = (UInt32)(5)
const CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = (UInt32)(6)
const CU_POINTER_ATTRIBUTE_BUFFER_ID = (UInt32)(7)
const CU_POINTER_ATTRIBUTE_IS_MANAGED = (UInt32)(8)
# end enum CUpointer_attribute_enum

# begin enum CUpointer_attribute
typealias CUpointer_attribute UInt32
const CU_POINTER_ATTRIBUTE_CONTEXT = (UInt32)(1)
const CU_POINTER_ATTRIBUTE_MEMORY_TYPE = (UInt32)(2)
const CU_POINTER_ATTRIBUTE_DEVICE_POINTER = (UInt32)(3)
const CU_POINTER_ATTRIBUTE_HOST_POINTER = (UInt32)(4)
const CU_POINTER_ATTRIBUTE_P2P_TOKENS = (UInt32)(5)
const CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = (UInt32)(6)
const CU_POINTER_ATTRIBUTE_BUFFER_ID = (UInt32)(7)
const CU_POINTER_ATTRIBUTE_IS_MANAGED = (UInt32)(8)
# end enum CUpointer_attribute

# begin enum CUfunction_attribute_enum
typealias CUfunction_attribute_enum UInt32
const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = (UInt32)(0)
const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = (UInt32)(1)
const CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = (UInt32)(2)
const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = (UInt32)(3)
const CU_FUNC_ATTRIBUTE_NUM_REGS = (UInt32)(4)
const CU_FUNC_ATTRIBUTE_PTX_VERSION = (UInt32)(5)
const CU_FUNC_ATTRIBUTE_BINARY_VERSION = (UInt32)(6)
const CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = (UInt32)(7)
const CU_FUNC_ATTRIBUTE_MAX = (UInt32)(8)
# end enum CUfunction_attribute_enum

# begin enum CUfunction_attribute
typealias CUfunction_attribute UInt32
const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = (UInt32)(0)
const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = (UInt32)(1)
const CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = (UInt32)(2)
const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = (UInt32)(3)
const CU_FUNC_ATTRIBUTE_NUM_REGS = (UInt32)(4)
const CU_FUNC_ATTRIBUTE_PTX_VERSION = (UInt32)(5)
const CU_FUNC_ATTRIBUTE_BINARY_VERSION = (UInt32)(6)
const CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = (UInt32)(7)
const CU_FUNC_ATTRIBUTE_MAX = (UInt32)(8)
# end enum CUfunction_attribute

# begin enum CUfunc_cache_enum
typealias CUfunc_cache_enum UInt32
const CU_FUNC_CACHE_PREFER_NONE = (UInt32)(0)
const CU_FUNC_CACHE_PREFER_SHARED = (UInt32)(1)
const CU_FUNC_CACHE_PREFER_L1 = (UInt32)(2)
const CU_FUNC_CACHE_PREFER_EQUAL = (UInt32)(3)
# end enum CUfunc_cache_enum

# begin enum CUfunc_cache
typealias CUfunc_cache UInt32
const CU_FUNC_CACHE_PREFER_NONE = (UInt32)(0)
const CU_FUNC_CACHE_PREFER_SHARED = (UInt32)(1)
const CU_FUNC_CACHE_PREFER_L1 = (UInt32)(2)
const CU_FUNC_CACHE_PREFER_EQUAL = (UInt32)(3)
# end enum CUfunc_cache

# begin enum CUsharedconfig_enum
typealias CUsharedconfig_enum UInt32
const CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = (UInt32)(0)
const CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = (UInt32)(1)
const CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = (UInt32)(2)
# end enum CUsharedconfig_enum

# begin enum CUsharedconfig
typealias CUsharedconfig UInt32
const CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = (UInt32)(0)
const CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = (UInt32)(1)
const CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = (UInt32)(2)
# end enum CUsharedconfig

# begin enum CUmemorytype_enum
typealias CUmemorytype_enum UInt32
const CU_MEMORYTYPE_HOST = (UInt32)(1)
const CU_MEMORYTYPE_DEVICE = (UInt32)(2)
const CU_MEMORYTYPE_ARRAY = (UInt32)(3)
const CU_MEMORYTYPE_UNIFIED = (UInt32)(4)
# end enum CUmemorytype_enum

# begin enum CUmemorytype
typealias CUmemorytype UInt32
const CU_MEMORYTYPE_HOST = (UInt32)(1)
const CU_MEMORYTYPE_DEVICE = (UInt32)(2)
const CU_MEMORYTYPE_ARRAY = (UInt32)(3)
const CU_MEMORYTYPE_UNIFIED = (UInt32)(4)
# end enum CUmemorytype

# begin enum CUcomputemode_enum
typealias CUcomputemode_enum UInt32
const CU_COMPUTEMODE_DEFAULT = (UInt32)(0)
const CU_COMPUTEMODE_PROHIBITED = (UInt32)(2)
const CU_COMPUTEMODE_EXCLUSIVE_PROCESS = (UInt32)(3)
# end enum CUcomputemode_enum

# begin enum CUcomputemode
typealias CUcomputemode UInt32
const CU_COMPUTEMODE_DEFAULT = (UInt32)(0)
const CU_COMPUTEMODE_PROHIBITED = (UInt32)(2)
const CU_COMPUTEMODE_EXCLUSIVE_PROCESS = (UInt32)(3)
# end enum CUcomputemode

# begin enum CUmem_advise_enum
typealias CUmem_advise_enum UInt32
const CU_MEM_ADVISE_SET_READ_MOSTLY = (UInt32)(1)
const CU_MEM_ADVISE_UNSET_READ_MOSTLY = (UInt32)(2)
const CU_MEM_ADVISE_SET_PREFERRED_LOCATION = (UInt32)(3)
const CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = (UInt32)(4)
const CU_MEM_ADVISE_SET_ACCESSED_BY = (UInt32)(5)
const CU_MEM_ADVISE_UNSET_ACCESSED_BY = (UInt32)(6)
# end enum CUmem_advise_enum

# begin enum CUmem_advise
typealias CUmem_advise UInt32
const CU_MEM_ADVISE_SET_READ_MOSTLY = (UInt32)(1)
const CU_MEM_ADVISE_UNSET_READ_MOSTLY = (UInt32)(2)
const CU_MEM_ADVISE_SET_PREFERRED_LOCATION = (UInt32)(3)
const CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = (UInt32)(4)
const CU_MEM_ADVISE_SET_ACCESSED_BY = (UInt32)(5)
const CU_MEM_ADVISE_UNSET_ACCESSED_BY = (UInt32)(6)
# end enum CUmem_advise

# begin enum CUmem_range_attribute_enum
typealias CUmem_range_attribute_enum UInt32
const CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = (UInt32)(1)
const CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = (UInt32)(2)
const CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = (UInt32)(3)
const CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = (UInt32)(4)
# end enum CUmem_range_attribute_enum

# begin enum CUmem_range_attribute
typealias CUmem_range_attribute UInt32
const CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = (UInt32)(1)
const CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = (UInt32)(2)
const CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = (UInt32)(3)
const CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = (UInt32)(4)
# end enum CUmem_range_attribute

# begin enum CUjit_option_enum
typealias CUjit_option_enum UInt32
const CU_JIT_MAX_REGISTERS = (UInt32)(0)
const CU_JIT_THREADS_PER_BLOCK = (UInt32)(1)
const CU_JIT_WALL_TIME = (UInt32)(2)
const CU_JIT_INFO_LOG_BUFFER = (UInt32)(3)
const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = (UInt32)(4)
const CU_JIT_ERROR_LOG_BUFFER = (UInt32)(5)
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = (UInt32)(6)
const CU_JIT_OPTIMIZATION_LEVEL = (UInt32)(7)
const CU_JIT_TARGET_FROM_CUCONTEXT = (UInt32)(8)
const CU_JIT_TARGET = (UInt32)(9)
const CU_JIT_FALLBACK_STRATEGY = (UInt32)(10)
const CU_JIT_GENERATE_DEBUG_INFO = (UInt32)(11)
const CU_JIT_LOG_VERBOSE = (UInt32)(12)
const CU_JIT_GENERATE_LINE_INFO = (UInt32)(13)
const CU_JIT_CACHE_MODE = (UInt32)(14)
const CU_JIT_NEW_SM3X_OPT = (UInt32)(15)
const CU_JIT_FAST_COMPILE = (UInt32)(16)
const CU_JIT_NUM_OPTIONS = (UInt32)(17)
# end enum CUjit_option_enum

# begin enum CUjit_option
typealias CUjit_option UInt32
const CU_JIT_MAX_REGISTERS = (UInt32)(0)
const CU_JIT_THREADS_PER_BLOCK = (UInt32)(1)
const CU_JIT_WALL_TIME = (UInt32)(2)
const CU_JIT_INFO_LOG_BUFFER = (UInt32)(3)
const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = (UInt32)(4)
const CU_JIT_ERROR_LOG_BUFFER = (UInt32)(5)
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = (UInt32)(6)
const CU_JIT_OPTIMIZATION_LEVEL = (UInt32)(7)
const CU_JIT_TARGET_FROM_CUCONTEXT = (UInt32)(8)
const CU_JIT_TARGET = (UInt32)(9)
const CU_JIT_FALLBACK_STRATEGY = (UInt32)(10)
const CU_JIT_GENERATE_DEBUG_INFO = (UInt32)(11)
const CU_JIT_LOG_VERBOSE = (UInt32)(12)
const CU_JIT_GENERATE_LINE_INFO = (UInt32)(13)
const CU_JIT_CACHE_MODE = (UInt32)(14)
const CU_JIT_NEW_SM3X_OPT = (UInt32)(15)
const CU_JIT_FAST_COMPILE = (UInt32)(16)
const CU_JIT_NUM_OPTIONS = (UInt32)(17)
# end enum CUjit_option

# begin enum CUjit_target_enum
typealias CUjit_target_enum UInt32
const CU_TARGET_COMPUTE_10 = (UInt32)(10)
const CU_TARGET_COMPUTE_11 = (UInt32)(11)
const CU_TARGET_COMPUTE_12 = (UInt32)(12)
const CU_TARGET_COMPUTE_13 = (UInt32)(13)
const CU_TARGET_COMPUTE_20 = (UInt32)(20)
const CU_TARGET_COMPUTE_21 = (UInt32)(21)
const CU_TARGET_COMPUTE_30 = (UInt32)(30)
const CU_TARGET_COMPUTE_32 = (UInt32)(32)
const CU_TARGET_COMPUTE_35 = (UInt32)(35)
const CU_TARGET_COMPUTE_37 = (UInt32)(37)
const CU_TARGET_COMPUTE_50 = (UInt32)(50)
const CU_TARGET_COMPUTE_52 = (UInt32)(52)
const CU_TARGET_COMPUTE_53 = (UInt32)(53)
const CU_TARGET_COMPUTE_60 = (UInt32)(60)
const CU_TARGET_COMPUTE_61 = (UInt32)(61)
const CU_TARGET_COMPUTE_62 = (UInt32)(62)
# end enum CUjit_target_enum

# begin enum CUjit_target
typealias CUjit_target UInt32
const CU_TARGET_COMPUTE_10 = (UInt32)(10)
const CU_TARGET_COMPUTE_11 = (UInt32)(11)
const CU_TARGET_COMPUTE_12 = (UInt32)(12)
const CU_TARGET_COMPUTE_13 = (UInt32)(13)
const CU_TARGET_COMPUTE_20 = (UInt32)(20)
const CU_TARGET_COMPUTE_21 = (UInt32)(21)
const CU_TARGET_COMPUTE_30 = (UInt32)(30)
const CU_TARGET_COMPUTE_32 = (UInt32)(32)
const CU_TARGET_COMPUTE_35 = (UInt32)(35)
const CU_TARGET_COMPUTE_37 = (UInt32)(37)
const CU_TARGET_COMPUTE_50 = (UInt32)(50)
const CU_TARGET_COMPUTE_52 = (UInt32)(52)
const CU_TARGET_COMPUTE_53 = (UInt32)(53)
const CU_TARGET_COMPUTE_60 = (UInt32)(60)
const CU_TARGET_COMPUTE_61 = (UInt32)(61)
const CU_TARGET_COMPUTE_62 = (UInt32)(62)
# end enum CUjit_target

# begin enum CUjit_fallback_enum
typealias CUjit_fallback_enum UInt32
const CU_PREFER_PTX = (UInt32)(0)
const CU_PREFER_BINARY = (UInt32)(1)
# end enum CUjit_fallback_enum

# begin enum CUjit_fallback
typealias CUjit_fallback UInt32
const CU_PREFER_PTX = (UInt32)(0)
const CU_PREFER_BINARY = (UInt32)(1)
# end enum CUjit_fallback

# begin enum CUjit_cacheMode_enum
typealias CUjit_cacheMode_enum UInt32
const CU_JIT_CACHE_OPTION_NONE = (UInt32)(0)
const CU_JIT_CACHE_OPTION_CG = (UInt32)(1)
const CU_JIT_CACHE_OPTION_CA = (UInt32)(2)
# end enum CUjit_cacheMode_enum

# begin enum CUjit_cacheMode
typealias CUjit_cacheMode UInt32
const CU_JIT_CACHE_OPTION_NONE = (UInt32)(0)
const CU_JIT_CACHE_OPTION_CG = (UInt32)(1)
const CU_JIT_CACHE_OPTION_CA = (UInt32)(2)
# end enum CUjit_cacheMode

# begin enum CUjitInputType_enum
typealias CUjitInputType_enum UInt32
const CU_JIT_INPUT_CUBIN = (UInt32)(0)
const CU_JIT_INPUT_PTX = (UInt32)(1)
const CU_JIT_INPUT_FATBINARY = (UInt32)(2)
const CU_JIT_INPUT_OBJECT = (UInt32)(3)
const CU_JIT_INPUT_LIBRARY = (UInt32)(4)
const CU_JIT_NUM_INPUT_TYPES = (UInt32)(5)
# end enum CUjitInputType_enum

# begin enum CUjitInputType
typealias CUjitInputType UInt32
const CU_JIT_INPUT_CUBIN = (UInt32)(0)
const CU_JIT_INPUT_PTX = (UInt32)(1)
const CU_JIT_INPUT_FATBINARY = (UInt32)(2)
const CU_JIT_INPUT_OBJECT = (UInt32)(3)
const CU_JIT_INPUT_LIBRARY = (UInt32)(4)
const CU_JIT_NUM_INPUT_TYPES = (UInt32)(5)
# end enum CUjitInputType

typealias CUlinkState_st Void
typealias CUlinkState Ptr{CUlinkState_st}

# begin enum CUgraphicsRegisterFlags_enum
typealias CUgraphicsRegisterFlags_enum UInt32
const CU_GRAPHICS_REGISTER_FLAGS_NONE = (UInt32)(0)
const CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = (UInt32)(1)
const CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = (UInt32)(2)
const CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = (UInt32)(4)
const CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = (UInt32)(8)
# end enum CUgraphicsRegisterFlags_enum

# begin enum CUgraphicsRegisterFlags
typealias CUgraphicsRegisterFlags UInt32
const CU_GRAPHICS_REGISTER_FLAGS_NONE = (UInt32)(0)
const CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = (UInt32)(1)
const CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = (UInt32)(2)
const CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = (UInt32)(4)
const CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = (UInt32)(8)
# end enum CUgraphicsRegisterFlags

# begin enum CUgraphicsMapResourceFlags_enum
typealias CUgraphicsMapResourceFlags_enum UInt32
const CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = (UInt32)(0)
const CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = (UInt32)(1)
const CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = (UInt32)(2)
# end enum CUgraphicsMapResourceFlags_enum

# begin enum CUgraphicsMapResourceFlags
typealias CUgraphicsMapResourceFlags UInt32
const CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = (UInt32)(0)
const CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = (UInt32)(1)
const CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = (UInt32)(2)
# end enum CUgraphicsMapResourceFlags

# begin enum CUarray_cubemap_face_enum
typealias CUarray_cubemap_face_enum UInt32
const CU_CUBEMAP_FACE_POSITIVE_X = (UInt32)(0)
const CU_CUBEMAP_FACE_NEGATIVE_X = (UInt32)(1)
const CU_CUBEMAP_FACE_POSITIVE_Y = (UInt32)(2)
const CU_CUBEMAP_FACE_NEGATIVE_Y = (UInt32)(3)
const CU_CUBEMAP_FACE_POSITIVE_Z = (UInt32)(4)
const CU_CUBEMAP_FACE_NEGATIVE_Z = (UInt32)(5)
# end enum CUarray_cubemap_face_enum

# begin enum CUarray_cubemap_face
typealias CUarray_cubemap_face UInt32
const CU_CUBEMAP_FACE_POSITIVE_X = (UInt32)(0)
const CU_CUBEMAP_FACE_NEGATIVE_X = (UInt32)(1)
const CU_CUBEMAP_FACE_POSITIVE_Y = (UInt32)(2)
const CU_CUBEMAP_FACE_NEGATIVE_Y = (UInt32)(3)
const CU_CUBEMAP_FACE_POSITIVE_Z = (UInt32)(4)
const CU_CUBEMAP_FACE_NEGATIVE_Z = (UInt32)(5)
# end enum CUarray_cubemap_face

# begin enum CUlimit_enum
typealias CUlimit_enum UInt32
const CU_LIMIT_STACK_SIZE = (UInt32)(0)
const CU_LIMIT_PRINTF_FIFO_SIZE = (UInt32)(1)
const CU_LIMIT_MALLOC_HEAP_SIZE = (UInt32)(2)
const CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = (UInt32)(3)
const CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = (UInt32)(4)
const CU_LIMIT_MAX = (UInt32)(5)
# end enum CUlimit_enum

# begin enum CUlimit
typealias CUlimit UInt32
const CU_LIMIT_STACK_SIZE = (UInt32)(0)
const CU_LIMIT_PRINTF_FIFO_SIZE = (UInt32)(1)
const CU_LIMIT_MALLOC_HEAP_SIZE = (UInt32)(2)
const CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = (UInt32)(3)
const CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = (UInt32)(4)
const CU_LIMIT_MAX = (UInt32)(5)
# end enum CUlimit

# begin enum CUresourcetype_enum
typealias CUresourcetype_enum UInt32
const CU_RESOURCE_TYPE_ARRAY = (UInt32)(0)
const CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = (UInt32)(1)
const CU_RESOURCE_TYPE_LINEAR = (UInt32)(2)
const CU_RESOURCE_TYPE_PITCH2D = (UInt32)(3)
# end enum CUresourcetype_enum

# begin enum CUresourcetype
typealias CUresourcetype UInt32
const CU_RESOURCE_TYPE_ARRAY = (UInt32)(0)
const CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = (UInt32)(1)
const CU_RESOURCE_TYPE_LINEAR = (UInt32)(2)
const CU_RESOURCE_TYPE_PITCH2D = (UInt32)(3)
# end enum CUresourcetype

# begin enum cudaError_enum
typealias cudaError_enum UInt32
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
const CUDA_ERROR_NVLINK_UNCORRECTABLE = (UInt32)(220)
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
typealias CUresult UInt32
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
const CUDA_ERROR_NVLINK_UNCORRECTABLE = (UInt32)(220)
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

# begin enum CUdevice_P2PAttribute_enum
typealias CUdevice_P2PAttribute_enum UInt32
const CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = (UInt32)(1)
const CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = (UInt32)(2)
const CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = (UInt32)(3)
# end enum CUdevice_P2PAttribute_enum

# begin enum CUdevice_P2PAttribute
typealias CUdevice_P2PAttribute UInt32
const CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = (UInt32)(1)
const CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = (UInt32)(2)
const CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = (UInt32)(3)
# end enum CUdevice_P2PAttribute

typealias CUstreamCallback Ptr{Void}
typealias CUoccupancyB2DSize Ptr{Void}

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

zero(::Type{Array_3_CUaddress_mode}) = begin  # /home/shindo/local-lemon/.julia/v0.4/Clang/src/wrap_c.jl, line 266:
        Array_3_CUaddress_mode(fill(zero(CUaddress_mode),3)...)
    end

immutable Array_4_Cfloat
    d1::Cfloat
    d2::Cfloat
    d3::Cfloat
    d4::Cfloat
end

zero(::Type{Array_4_Cfloat}) = begin  # /home/shindo/local-lemon/.julia/v0.4/Clang/src/wrap_c.jl, line 266:
        Array_4_Cfloat(fill(zero(Cfloat),4)...)
    end

immutable Array_12_Cint
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
end

zero(::Type{Array_12_Cint}) = begin  # /home/shindo/local-lemon/.julia/v0.4/Clang/src/wrap_c.jl, line 266:
        Array_12_Cint(fill(zero(Cint),12)...)
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
    borderColor::Array_4_Cfloat
    reserved::Array_12_Cint
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
    borderColor::Array_4_Cfloat
    reserved::Array_12_Cint
end

# begin enum CUresourceViewFormat_enum
typealias CUresourceViewFormat_enum UInt32
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
typealias CUresourceViewFormat UInt32
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

zero(::Type{Array_16_UInt32}) = begin  # /home/shindo/local-lemon/.julia/v0.4/Clang/src/wrap_c.jl, line 266:
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
