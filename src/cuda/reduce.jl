import Base: sum, findmax, findmin, maximum, argmax
import LinearAlgebra: norm
export average

function sum(x::CuArray; dims=())
    if isempty(dims)
        x = vec(x)
        dims = (1,)
    else
        isa(dims,Int) && (dims = (dims,))
    end
    for d in dims
        x = CUDNN.reduce(x, d, CUDNN.CUDNN_REDUCE_TENSOR_ADD)[1]
    end
    x
end

function argmax(x::CuArray, dims::Int)
    findmax(x, dims=dims)[2]
end

function findmax(x::CuArray; dims::Int)
    dim = dims
    y, idx = CUDNN.reduce(x, dim, CUDNN.CUDNN_REDUCE_TENSOR_MAX)
    idx = reshape(idx, Base.setindex(size(x),1,dim))
    y, idx
end

function findmin(x::CuArray; dims::Int)
    CUDNN.reduce(x, dims, CUDNN.CUDNN_REDUCE_TENSOR_MIN)
end

average(x::CuArray; dims::Int) = CUDNN.reduce(x, dims, CUDNN.CUDNN_REDUCE_TENSOR_AVG)[1]

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
