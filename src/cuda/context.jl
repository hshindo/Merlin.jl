const CuContext_t = Ptr{Void}
const CuContexts = CuContext_t[]

function init_contexts()
    for dev = 0:ndevices()-1
        ref = Ref{CuContext_t}()
        @apicall :cuCtxCreate (Ptr{CuContext_t},Cuint,Cint) ref 0 dev
        push!(CuContexts, ref[])

        cap = capability(dev)
        mem = round(Int, totalmem(dev) / (1024^2))
        info("device[$dev]: $(devicename(dev)), capability $(cap[1]).$(cap[2]), totalmem = $(mem) MB")
    end
    setdevice(0)
    # @apicall :cuCtxDestroy (CuContext_t,) ctx
end

function getdevice()
    ref = Ref{Cint}()
    @apicall :cuCtxGetDevice (Ptr{Cint},) ref
    Int(ref[])
end

function setdevice(dev::Int)
    @apicall :cuCtxSetCurrent (CuContext_t,) CuContexts[dev+1]
end

function synchronize()
    @apicall :cuCtxSynchronize ()
end
