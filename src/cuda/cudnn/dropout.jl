mutable struct DropoutDesc
    ptr::Cptr
    reserve_space

    function DropoutDesc(droprate::Float64, seed::Int=0)
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateDropoutDescriptor (Ptr{Cptr},) ref
        ptr = ref[]
        desc = new(ptr, nothing)

        h = gethandle()
        ref = Ref{Csize_t}()
        @cudnn :cudnnDropoutGetStatesSize (Cptr,Ptr{Csize_t}) h ref
        states = CuArray{UInt8}(Int(ref[]))
        @cudnn(:cudnnSetDropoutDescriptor,
            (Cptr,Cptr,Cfloat,Cptr,Csize_t,Culonglong),
            desc, h, droprate, states, length(states), seed)

        finalizer(desc, x -> @cudnn :cudnnDestroyDropoutDescriptor (Cptr,) x.ptr)
        desc
    end
end

Base.unsafe_convert(::Type{Cptr}, desc::DropoutDesc) = desc.ptr

function dropout(x::CuArray{T,N}, droprate::Float64) where {T,N}
    dropdesc = DropoutDesc(droprate)
    xdesc = TensorDesc(x, 4)

    ref = Ref{Csize_t}()
    @cudnn :cudnnDropoutGetReserveSpaceSize (Cptr,Ptr{Csize_t}) xdesc ref
    reserve_space = CuArray{UInt8}(Int(ref[]))
    dropdesc.reserve_space = reserve_space

    h = gethandle()
    y = similar(x)
    @cudnn(:cudnnDropoutForward,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Csize_t),
        h, dropdesc, xdesc, x, xdesc, y, reserve_space, length(reserve_space))

    y, dropdesc
end

function ∇dropout!(dy::CuArray, dx::CuArray, droprate, dropdesc)
    xdesc = TensorDesc(dy, 4)
    h = gethandle()
    reserve_space = dropdesc.reserve_space
    @cudnn(:cudnnDropoutBackward,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Csize_t),
        h, dropdesc, xdesc, dy, xdesc, dx, reserve_space, length(reserve_space))
end
