mutable struct DropoutDesc
    ptr::Cptr
    states

    function DropoutDesc(droprate::Float64)
        seed = 0
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateDropoutDescriptor (Ptr{Cptr},) ref
        desc = new(ref[])

        h = gethandle()
        ref = Ref{Csize_t}()
        @cudnn :cudnnDropoutGetStatesSize (Cptr,Ptr{Csize_t}) h ref
        states = CuArray{UInt8}(Int(ref[]))
        @cudnn(:cudnnSetDropoutDescriptor,
            (Cptr,Cptr,Cfloat,Cptr,Csize_t,Culonglong),
            desc, h, droprate, states, length(states), seed)
        desc.states = states

        finalizer(desc) do x
            @cudnn :cudnnDestroyDropoutDescriptor (Cptr,) x.ptr
        end
        desc
    end
end

Base.cconvert(::Type{Cptr}, desc::DropoutDesc) = desc.ptr

const DICT_DropoutDesc = Dict()

function dropout(x::CuArray{T,N}, droprate::Float64) where {T,N}
    h = gethandle()
    dropdesc = get!(DICT_DropoutDesc, (h,droprate)) do
        DropoutDesc(droprate)
    end
    xdesc = TensorDesc(x, 4)

    ref = Ref{Csize_t}()
    @cudnn :cudnnDropoutGetReserveSpaceSize (Cptr,Ptr{Csize_t}) xdesc ref
    reserve_space = CuArray{UInt8}(Int(ref[]))

    y = similar(x)
    @cudnn(:cudnnDropoutForward,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Csize_t),
        h, dropdesc, xdesc, x, xdesc, y, reserve_space, length(reserve_space))
    y, (dropdesc,xdesc,reserve_space)
end

function ∇dropout!(dy::CuArray, dx::CuArray, work)
    dropdesc, xdesc, reserve_space = work
    h = gethandle()
    @cudnn(:cudnnDropoutBackward,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Csize_t),
        h, dropdesc, xdesc, dy, xdesc, dx, reserve_space, length(reserve_space))
end
