function getdevice()
    ref = Ref{Cint}()
    @apicall :cuCtxGetDevice (Ptr{Cint},) ref
    Int(ref[])
end

function setdevice(dev::Int)
    #cap = capability(dev)
    #mem = round(Int, totalmem(dev) / (1024^2))
    #@info "device[$dev]: $(devicename(dev)), capability $(cap[1]).$(cap[2]), totalmem = $(mem) MB"
    getdevice() == dev && return
    isassigned(CONTEXTS,dev) || (CONTEXTS[dev+1] = CuContext(dev))
    setcontext(CONTEXTS[dev+1])
    dev
end
function setdevice(f::Function, dev::Int)
    _dev = getdevice()
    dev == _dev || setdevice(dev)
    f()
    dev == _dev || setdevice(_dev)
end

function ndevices()
    ref = Ref{Cint}()
    @apicall :cuDeviceGetCount (Ptr{Cint},) ref
    Int(ref[])
end

function devicename(dev::Int)
    buflen = 256
    buf = Vector{Cchar}(buflen)
    @apicall :cuDeviceGetName (Ptr{Cchar},Cint,Cint) buf buflen dev
    buf[end] = 0
    unsafe_string(pointer(buf))
end

function totalmem(dev::Int)
    ref = Ref{Csize_t}()
    @apicall :cuDeviceTotalMem (Ptr{Csize_t},Cint) ref dev
    Int(ref[])
end

@enum(CUdevice_attribute, MAX_THREADS_PER_BLOCK = Cint(1),
                          MAX_BLOCK_DIM_X,
                          MAX_BLOCK_DIM_Y,
                          MAX_BLOCK_DIM_Z,
                          MAX_GRID_DIM_X,
                          MAX_GRID_DIM_Y,
                          MAX_GRID_DIM_Z,
                          MAX_SHARED_MEMORY_PER_BLOCK,
                          TOTAL_CONSTANT_MEMORY,
                          WARP_SIZE,
                          MAX_PITCH,
                          MAX_REGISTERS_PER_BLOCK,
                          CLOCK_RATE,
                          TEXTURE_ALIGNMENT,
                          GPU_OVERLAP,
                          MULTIPROCESSOR_COUNT,
                          KERNEL_EXEC_TIMEOUT,
                          INTEGRATED,
                          CAN_MAP_HOST_MEMORY,
                          COMPUTE_MODE,
                          MAXIMUM_TEXTURE1D_WIDTH,
                          MAXIMUM_TEXTURE2D_WIDTH,
                          MAXIMUM_TEXTURE2D_HEIGHT,
                          MAXIMUM_TEXTURE3D_WIDTH,
                          MAXIMUM_TEXTURE3D_HEIGHT,
                          MAXIMUM_TEXTURE3D_DEPTH,
                          MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
                          MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
                          MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
                          SURFACE_ALIGNMENT,
                          CONCURRENT_KERNELS,
                          ECC_ENABLED,
                          PCI_BUS_ID,
                          PCI_DEVICE_ID,
                          TCC_DRIVER,
                          MEMORY_CLOCK_RATE,
                          GLOBAL_MEMORY_BUS_WIDTH,
                          L2_CACHE_SIZE,
                          MAX_THREADS_PER_MULTIPROCESSOR,
                          ASYNC_ENGINE_COUNT,
                          UNIFIED_ADDRESSING,
                          MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
                          MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
                          CAN_TEX2D_GATHER,
                          MAXIMUM_TEXTURE2D_GATHER_WIDTH,
                          MAXIMUM_TEXTURE2D_GATHER_HEIGHT,
                          MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE,
                          MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE,
                          MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE,
                          PCI_DOMAIN_ID,
                          TEXTURE_PITCH_ALIGNMENT,
                          MAXIMUM_TEXTURECUBEMAP_WIDTH,
                          MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH,
                          MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS,
                          MAXIMUM_SURFACE1D_WIDTH,
                          MAXIMUM_SURFACE2D_WIDTH,
                          MAXIMUM_SURFACE2D_HEIGHT,
                          MAXIMUM_SURFACE3D_WIDTH,
                          MAXIMUM_SURFACE3D_HEIGHT,
                          MAXIMUM_SURFACE3D_DEPTH,
                          MAXIMUM_SURFACE1D_LAYERED_WIDTH,
                          MAXIMUM_SURFACE1D_LAYERED_LAYERS,
                          MAXIMUM_SURFACE2D_LAYERED_WIDTH,
                          MAXIMUM_SURFACE2D_LAYERED_HEIGHT,
                          MAXIMUM_SURFACE2D_LAYERED_LAYERS,
                          MAXIMUM_SURFACECUBEMAP_WIDTH,
                          MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH,
                          MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS,
                          MAXIMUM_TEXTURE1D_LINEAR_WIDTH,
                          MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
                          MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
                          MAXIMUM_TEXTURE2D_LINEAR_PITCH,
                          MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH,
                          MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT,
                          COMPUTE_CAPABILITY_MAJOR,
                          COMPUTE_CAPABILITY_MINOR,
                          MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH,
                          STREAM_PRIORITIES_SUPPORTED,
                          GLOBAL_L1_CACHE_SUPPORTED,
                          LOCAL_L1_CACHE_SUPPORTED,
                          MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                          MAX_REGISTERS_PER_MULTIPROCESSOR,
                          MANAGED_MEMORY,
                          MULTI_GPU_BOARD,
                          MULTI_GPU_BOARD_GROUP_ID)
@assert Cint(MULTI_GPU_BOARD_GROUP_ID) == 85

function attribute(dev::Int, code::CUdevice_attribute)
    ref = Ref{Cint}()
    @apicall :cuDeviceGetAttribute (Ptr{Cint},Cint,Cint) ref code dev
    Int(ref[])
end

function capability(dev::Int)
    major = attribute(dev, COMPUTE_CAPABILITY_MAJOR)
    minor = attribute(dev, COMPUTE_CAPABILITY_MINOR)
    major, minor
end

warpsize(dev::Int) = attribute(dev, WARP_SIZE)
