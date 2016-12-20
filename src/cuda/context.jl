export
    device, setdevice,
    synchronize

const contexts = Ptr{Void}[]

function initctx()
    cuInit(0)
    for dev = 0:devcount()-1
        ref = Ptr{Void}[0]
        cuCtxCreate(ref, 0, dev)
        push!(contexts, ref[1])
    end
    setdevice(0)
end

function device()
    ref = Ptr{Void}[0]
    cuCtxGetDevice(ref)
    Int(ref[1])
end

setdevice(dev::Int) = cuCtxSetCurrent(contexts[dev+1])

synchronize() = cuCtxSynchronize()
