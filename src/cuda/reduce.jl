import Base: sum, findmax, findmin, maximum
import Statistics: mean
import LinearAlgebra: norm

sum(x::CuArray; dims::Int) = CUDNN.reduce(x, dims, CUDNN.CUDNN_REDUCE_TENSOR_ADD)[1]
function sum(x::CuArray)
    x = vec(x)
    Array(sum(x,1))[1]
end

findmax(x::CuArray; dims::Int) = CUDNN.reduce(x, dims, CUDNN.CUDNN_REDUCE_TENSOR_MAX)

findmin(x::CuArray; dims::Int) = CUDNN.reduce(x, dims, CUDNN.CUDNN_REDUCE_TENSOR_MIN)

# maximum(::typeof(abs), x::CuArray, dim::Int) = CUDNN.reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_AMAX)[1]

mean(x::CuArray; dims::Int) = CUDNN.reduce(x, dims, CUDNN.CUDNN_REDUCE_TENSOR_AVG)[1]

function norm(x::CuArray, dim::Int, p::Int)
    if p == 1
        CUDNN.reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_NORM1)[1]
    elseif p == 2
        CUDNN.reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_NORM2)[1]
    else
        throw("Not supported. Valid p: 1 or 2.")
    end
end

#=
function argmax(x::Array, dim::Int)
    _, index = findmax(x, dim)
    y = ind2sub(size(x), vec(index))[dim]
    dims = ntuple(i -> i==dim ? 1 : size(x,i), ndims(x))
    reshape(y, dims)
end
=#
