function reduce(A::CuArray{T}, dim::Int, op) where T
    # C = alpha * reduce_op(A) + beta * C
    reducedesc = CUDNN.ReduceTensorDesc(T, op)
    h = CUDNN.HANDLE
    adesc = CUDNN.TensorDesc(A)
    cdims = ntuple(ndims(A)) do i
        i == dim ? 1 : size(A,i)
    end
    C = similar(A, cdims)
    cdesc = CUDNN.TensorDesc(C)

    p = Cint[0]
    CUDNN.cudnnGetReductionIndicesSize(h, reducedesc, adesc, cdesc, p)
    indices_size = p[1]
    indices = indices_size == 0 ? C_NULL : CuArray{Int8}(Int(indices_size))

    p = Cint[0]
    CUDNN.cudnnGetReductionWorkspaceSize(h, reduce_desc, adesc, cdesc, p)
    workspace_size = p[1]
    workspace = CuArray{Int8}(Int(workspace_size))

    CUDNN.cudnnReduceTensor(h, reduce_desc, indices, indices_size, workspace, workspace_size, T[1], adesc, A, T[0], cdesc, C)
    indices == C_NULL ? C : (C,indices)
end

Base.sum(x::CuArray, dim) = reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_ADD)
mul(x::CuArray, dim) = reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_MUL)
Base.findmax(x::CuArray, dim) = reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_MIN)
Base.findmin(x::CuArray, dim) = reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_MAX)
# maxabs(x::CuArray, dim) = reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_AMAX)
Base.mean(x::CuArray, dim) = reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_AVG)
function Base.norm(x::CuArray, dim, p::Int)
    if p == 1
        reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_NORM1)
    elseif p == 2
        reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_NORM2)
    else
        throw("Not supported in CUDNN.")
    end
end
mul_nozeros(x::CuArray, dim) = reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS)

function ∇maximum!(gy::CuDeviceArray, gx::CuDeviceArray, idx::CuDeviceArray)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    gx[idx[i]] += gy[i]
    return nothing
end
