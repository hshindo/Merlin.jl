export Kernel, cudims
const DEVICE_H = String(open(read, joinpath(@__DIR__,"device.h")))

mutable struct Kernel
    ptx::String
    funs::Vector{CuFunction}
end

function Kernel(kernel::String)
    kernel = "$DEVICE_H\n$kernel"
    ptx = NVRTC.compile(kernel)
    funs = Array{CuFunction}(undef, nthreads()*ndevices())
    Kernel(ptx, funs)
end

function (k::Kernel)(griddims, blockdims, args...; sharedmem=0, stream=C_NULL)
    id = getdevice() * nthreads() + threadid()
    if !isassigned(k.funs, id)
        k.funs[id] = CuFunction(k.ptx)
    end
    f = k.funs[id]

    argptrs = [cubox(args[i]) for i=1:length(args)]
    @apicall(:cuLaunchKernel, (
        Ptr{Cvoid},           # function
        Cuint,Cuint,Cuint,      # grid dimensions (x, y, z)
        Cuint,Cuint,Cuint,      # block dimensions (x, y, z)
        Cuint,                  # shared memory bytes,
        Ptr{Cvoid},             # stream
        Ptr{Ptr{Cvoid}},         # kernel parameters
        Ptr{Ptr{Cvoid}}),         # extra parameters
        f,
        griddims[1], griddims[2], griddims[3],
        blockdims[1], blockdims[2], blockdims[3],
        sharedmem, stream, pointer(argptrs), C_NULL)
end

function cudims(n::Int)
    bx = 256
    gx = n <= bx ? 1 : ceil(Int, n/bx)
    (gx,1,1), (bx,1,1)
end

cubox(x::Int) = cubox(Cint(x))
cubox(x::Ptr) = x
cubox(x::CuPtr) = x.ptr
cubox(x::Ref) = x
cubox(x) = Ref(x)
cubox(x::CuArray) = cubox(CuDeviceArray(x))
cubox(x::CuSubArray) = cubox(CuDeviceArray(x))
