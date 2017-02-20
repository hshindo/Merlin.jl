const contexts = Ptr{Void}[]
function initctx()
    cuInit(0)
    for dev = 0:devcount()-1
        p = Ptr{Void}[0]
        cuCtxCreate(p, 0, dev)
        push!(contexts, p[1])
    end
    setdevice(0)
end

function infodevices()
    count = devcount()
    if count == 0
        println("No CUDA-capable device found.")
    else
        for dev = 0:count-1
            name = devname(dev)
            cap = capability(dev)
            mem = round(Int, totalmem(dev) / (1024^2))
            info("device[$dev]: $(name), capability $(cap[1]).$(cap[2]), totalmem = $(mem) MB")
        end
    end
end

function capability(dev::Int)
    major, minor = Cint[0], Cint[0]
    cuDeviceComputeCapability(major, minor, dev)
    Int(major[1]), Int(minor[1])
end

function device()
    p = Ptr{Void}[0]
    cuCtxGetDevice(p)
    Int(p[1])
end

function devcount()
    p = Cint[0]
    cuDeviceGetCount(p)
    Int(p[1])
end

function devname(dev::Int)
    p = Array{UInt8}(256)
    cuDeviceGetName(p, length(p), dev)
    unsafe_string(pointer(p))
end

function totalmem(dev::Int)
    p = Csize_t[0]
    cuDeviceTotalMem(p, dev)
    Int(p[1])
end

setdevice(dev::Int) = cuCtxSetCurrent(contexts[dev+1])

synchronize() = cuCtxSynchronize()
