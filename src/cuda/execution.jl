cubox(x) = x
cubox(x::Int) = Cint(x)
cubox{N}(t::NTuple{N,Int}) = map(Cint, t)
cubox(x::Vector{Int}) = ntuple(i -> Cint(x[i]), length(x))

struct CuDim3
    x::Int
    y::Int
    z::Int
end

function launch(f::CuFunction, griddims::CuDim3, blockdims::CuDim3, args...; sharedmem=0, stream=C_NULL)
    argptrs = Ptr{Void}[pointer_from_objref(cubox(a)) for a in args]
    @apicall(:cuLaunchKernel, (
        CuFunction_t,           # function
        Cuint,Cuint,Cuint,      # grid dimensions (x, y, z)
        Cuint,Cuint,Cuint,      # block dimensions (x, y, z)
        Cuint,                  # shared memory bytes,
        CuStream_t,             # stream
        Ptr{Ptr{Void}},         # kernel parameters
        Ptr{Ptr{Void}}),         # extra parameters
        f,
        griddims.x, griddims.y, griddims.z,
        blockdims.x, blockdims.y, blockdims.z,
        sharedmem, stream, argptrs, C_NULL)
end
