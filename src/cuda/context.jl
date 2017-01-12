export
    device, setdevice,
    synchronize

const contexts = Ptr{Void}[]

function initctx()
    isempty(contexts) || throw("Context is not empty.")
    cuInit(0)
    for dev = 0:devcount()-1
        p = Ptr{Void}[0]
        cuCtxCreate(p, 0, dev)
        push!(contexts, p[1])
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
