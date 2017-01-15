export
    capability, devcount, devname, totalmem

function capability(dev::Int)
    major, minor = Cint[0], Cint[0]
    cuDeviceComputeCapability(major, minor, dev)
    Int(major[1]), Int(minor[1])
end

function devcount()
    ref = Cint[0]
    cuDeviceGetCount(ref)
    Int(ref[1])
end

function devname(dev::Int)
    ref = zeros(UInt8, 256)
    cuDeviceGetName(ref, length(ref), dev)
    unsafe_string(pointer(ref))
end

function totalmem(dev::Int)
    ref = Csize_t[0]
    cuDeviceTotalMem(ref, dev)
    Int(ref[1])
end

function attribute(attrib::Int, dev::Int)
    ref = Cint[0]
    cuDeviceGetAttribute(ref, attrib, dev)
    Int(ref[1])
end

function attributes(dev::Int)
    Dict(
    "MAX_THREADS_PER_BLOCK" => attribute(1, dev),
    "MAX_BLOCK_DIM_X" => attribute(2, dev),
    "MAX_BLOCK_DIM_Y" => attribute(3, dev),
    "MAX_BLOCK_DIM_Z" => attribute(4, dev),
    "MAX_GRID_DIM_X" => attribute(5, dev),
    "MAX_GRID_DIM_Y" => attribute(6, dev),
    "MAX_GRID_DIM_Z" => attribute(7, dev),
    "MAX_SHARED_MEMORY_PER_BLOCK" => attribute(8, dev),
    "TOTAL_CONSTANT_MEMORY" => attribute(9, dev),
    "WARP_SIZE" => attribute(10, dev),
    "MAX_PITCH" => attribute(11, dev),
    "MAX_REGISTERS_PER_BLOCK" => attribute(12, dev)
    )
end

function infodevices()
    count = devcount()
    if count == 0
        println("No CUDA-capable device found.")
        return
    end
    for dev = 0:count-1
        name = devname(dev)
        cap = capability(dev)
        mem = round(Int, totalmem(dev) / (1024^2))
        info("device[$dev]: $(name), capability $(cap[1]).$(cap[2]), totalmem = $(mem) MB")
    end
end
