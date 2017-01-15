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

function attribute(attr::Int, dev::Int)
    p = Cint[0]
    cudaDeviceGetAttribute(p, attr, dev)
    Int(p[1])
end

function capability(dev::Int)
    attribute(Int(cudaDevAttrComputeCapabilityMajor), dev),
    attribute(Int(cudaDevAttrComputeCapabilityMinor), dev)
end

function devcount()
    p = Cint[0]
    cudaGetDeviceCount(p)
    Int(p[1])
end

function device()
    p = Cint[0]
    cudaGetDevice(p)
    Int(p[1])
end

function properties(dev::Int)
    p = Array(cudaDeviceProp, 1)
    cudaGetDeviceProperties(p, dev)
    p[1]
end

function devname(dev::Int)
    prop = properties(dev)
    p = pointer([prop.name])
    unsafe_string(Ptr{UInt8}(p))
end

function totalmem(dev::Int)
    prop = properties(dev)
    Int(prop.totalGlobalMem)
end

setdevice(dev::Int) = cudaSetDevice(dev)

synchronize() = cudaDeviceSynchronize()
