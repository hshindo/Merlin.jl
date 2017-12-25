box(x) = x
box(x::Int) = Cint(x)
box{N}(t::NTuple{N,Int}) = map(Cint, t)
box(x::Vector{Int}) = ntuple(i -> Cint(x[i]), length(x))

function launch(f::CuFunction, args...;
    dx=1, dy=1, dz=1, bx=128, by=1, bz=1, sharedmem=0, stream=C_NULL)

    argptrs = Ptr{Void}[pointer_from_objref(box(a)) for a in args]
    gx = ceil(dx / bx)
    gy = ceil(dy / by)
    gz = ceil(dz / bz)
    cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedmem, stream, argptrs, stream)

    @apicall :cuLaunchKernel (
        CuFunction_t,           # function
        Cuint, Cuint, Cuint,    # grid dimensions (x, y, z)
        Cuint, Cuint, Cuint,    # block dimensions (x, y, z)
        Cuint,                  # shared memory bytes,
        CuStream_t,             # stream
        Ptr{Ptr{Void}},         # kernel parameters
        Ptr{Ptr{Void}}),        # extra parameters
        f,
        griddim.x, griddim.y, griddim.z,
        blockdim.x, blockdim.y, blockdim.z,
        shmem, stream, kernelParams, C_NULL)
    ) f gx gy gz bx by bz sharedmem stream
end

# Type alias for conveniently specifying the dimensions
# (e.g. `(len, 2)` instead of `CuDim3((len, 2))`)
const CuDim = Union{CuDim3,Int,Tuple{Int},Tuple{Int,Int},Tuple{Int,Int,Int}}

function launch(f::CuFunction, griddim::CuDim, blockdim::CuDim,
                        shmem::Int, stream::CuStream,
                        args...)
    griddim = CuDim3(griddim)
    blockdim = CuDim3(blockdim)
    (griddim.x>0 && griddim.y>0 && griddim.z>0)    || throw(ArgumentError("Grid dimensions should be non-null"))
    (blockdim.x>0 && blockdim.y>0 && blockdim.z>0) || throw(ArgumentError("Block dimensions should be non-null"))

    _launch(f, griddim, blockdim, shmem, stream, args...)
end

@generated function _launch(f::CuFunction, griddim::CuDim3, blockdim::CuDim3,
                            shmem::Int, stream::CuStream,
                            args::NTuple{N,Any}) where N
    all(isbits, args.parameters) || throw(ArgumentError("Arguments to kernel should be bitstype."))

    ex = Expr(:block)
    push!(ex.args, :(Base.@_inline_meta))

    # If f has N parameters, then kernelParams needs to be an array of N pointers.
    # Each of kernelParams[0] through kernelParams[N-1] must point to a region of memory
    # from which the actual kernel parameter will be copied.

    # put arguments in Ref boxes so that we can get a pointers to them
    arg_refs = Vector{Symbol}(uninitialized, N)
    for i in 1:N
        arg_refs[i] = gensym()
        push!(ex.args, :($(arg_refs[i]) = Base.RefValue(args[$i])))
    end

    # generate an array with pointers
    arg_ptrs = [:(Base.unsafe_convert(Ptr{Void}, $(arg_refs[i]))) for i in 1:N]
    push!(ex.args, :(kernelParams = [$(arg_ptrs...)]))

    # root the argument boxes to the array of pointers,
    # keeping them alive across the call to `cuLaunchKernel`
    if VERSION >= v"0.7.0-DEV.1850"
        push!(ex.args, :(Base.@gc_preserve $(arg_refs...) kernelParams))
    end

    push!(ex.args, :(
        @apicall(:cuLaunchKernel, (
            CuFunction_t,           # function
            Cuint, Cuint, Cuint,    # grid dimensions (x, y, z)
            Cuint, Cuint, Cuint,    # block dimensions (x, y, z)
            Cuint,                  # shared memory bytes,
            CuStream_t,             # stream
            Ptr{Ptr{Void}},         # kernel parameters
            Ptr{Ptr{Void}}),        # extra parameters
            f,
            griddim.x, griddim.y, griddim.z,
            blockdim.x, blockdim.y, blockdim.z,
            shmem, stream, kernelParams, C_NULL)
        )
    )

    return ex
end
