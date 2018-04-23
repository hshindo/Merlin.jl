Base.sum(x::CuArray, dim::Int) = CUDNN.reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_ADD)[1]
Base.findmax(x::CuArray, dim::Int) = CUDNN.reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_MAX)
Base.findmin(x::CuArray, dim::Int) = CUDNN.reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_MIN)
Base.maximum(::typeof(abs), x::CuArray, dim::Int) = CUDNN.reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_AMAX)[1]
Base.mean(x::CuArray, dim) = CUDNN.reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_AVG)[1]
function Base.norm(x::CuArray, dim::Int, p::Int)
    if p == 1
        CUDNN.reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_NORM1)[1]
    elseif p == 2
        CUDNN.reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_NORM2)[1]
    else
        throw("Not supported. Valid p: 1 or 2.")
    end
end

function Base.sum(x::CuArray{T}) where T
    # TODO: make this faster
    x = vec(x)
    Array(sum(x,1))[1]
    #ref = Ref{Ptr{Void}}()
    #bytesize = sizeof(T)
    #@apicall :cuMemAllocHost (Ptr{Ptr{Void}},Csize_t) ref bytesize
end
