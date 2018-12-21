# cudnnReduceTensorOp_t
const CUDNN_REDUCE_TENSOR_ADD = Cint(0)
const CUDNN_REDUCE_TENSOR_MUL = Cint(1)
const CUDNN_REDUCE_TENSOR_MIN = Cint(2)
const CUDNN_REDUCE_TENSOR_MAX = Cint(3)
const CUDNN_REDUCE_TENSOR_AMAX = Cint(4)
const CUDNN_REDUCE_TENSOR_AVG = Cint(5)
const CUDNN_REDUCE_TENSOR_NORM1 = Cint(6)
const CUDNN_REDUCE_TENSOR_NORM2 = Cint(7)
const CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = Cint(8)

# cudnnReduceTensorIndices_t
const CUDNN_REDUCE_TENSOR_NO_INDICES = Cint(0)
const CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = Cint(1)

mutable struct ReduceTensorDesc
    ptr::Cptr

    function ReduceTensorDesc(::Type{T}, op::Cint) where T
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateReduceTensorDescriptor (Ptr{Cptr},) ref
        desc = new(ref[])
        finalizer(desc) do x
            @cudnn :cudnnDestroyReduceTensorDescriptor (Cptr,) x.ptr
        end

        ind = op == CUDNN_REDUCE_TENSOR_MIN || op == CUDNN_REDUCE_TENSOR_MAX ?
            CUDNN_REDUCE_TENSOR_FLATTENED_INDICES :
            CUDNN_REDUCE_TENSOR_NO_INDICES
        @cudnn(:cudnnSetReduceTensorDescriptor,
            (Cptr,Cint,Cint,Cint,Cint,Cint),
            desc, op, datatype(T), CUDNN_NOT_PROPAGATE_NAN, ind, CUDNN.CUDNN_32BIT_INDICES)
        desc
    end
end

Base.cconvert(::Type{Cptr}, desc::ReduceTensorDesc) = desc.ptr

const DICT_ReduceTensorDesc = Dict()

function reduce(A::CuArray{T}, dim, op) where T
    if size(A,dim) == 1 # CUDNN_STATUS_BAD_PARAM
        C = A
        indices = fill!(CuArray{Cint}(length(A)÷size(A,dim)), 0)
        return C, indices
    end

    h = gethandle()
    reducedesc = get!(DICT_ReduceTensorDesc, (T,op)) do
        ReduceTensorDesc(T, op)
    end
    adesc = TensorDesc(A, 4)
    cdims = Int[size(A)...]
    cdims[dim] = 1
    C = similar(A, cdims...)
    cdesc = TensorDesc(C, 4)

    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetReductionIndicesSize,
        (Cptr,Cptr,Cptr,Cptr,Ptr{Csize_t}),
        h, reducedesc, adesc, cdesc, ref)
    indices = ref[] == 0 ? C_NULL : CuArray{Cint}(Int(ref[])÷sizeof(Cint))

    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetReductionWorkspaceSize,
        (Cptr,Cptr,Cptr,Cptr,Ptr{Csize_t}),
        h, reducedesc, adesc, cdesc, ref)
    workspace = ref[] == 0 ? C_NULL : CuArray{UInt8}(Int(ref[]))
    workspace_length = ref[]

    @cudnn(:cudnnReduceTensor,
        (Cptr,Cptr,Cptr,Csize_t,Cptr,Csize_t,
        Cptr,Cptr,Cptr,Cptr,Cptr,Cptr),
        h, reducedesc, indices, length(indices)*sizeof(Cint), workspace, workspace_length,
        T[1], adesc, A, T[0], cdesc, C)
    C, indices
end
