export BatchedArray
export batchsize, isconstsize, unsafe_split
import Base: +, -, *
import Base.LinAlg.BLAS: gemv, gemm, gemm!

immutable BatchedArray{T,N}
    data::Array{T,N}
    dims::Vector{Int}
end

BatchedVector{T} = BatchedArray{T,1}
BatchedMatrix{T} = BatchedArray{T,2}

function BatchedArray(arrays::Vector{Array{T,N}}) where {T,N}
    data = cat(N, arrays...)
    dims = map(x -> size(x,N), arrays)
    BatchedArray(data, dims)
end
BatchedArray(data::Array...) = BatchedArray([data...])

@inline Base.getindex(x::BatchedArray, key) = getindex(x.data, key)
@inline Base.setindex!(x::BatchedArray, value, key) = setindex!(x.data, value, key)
@inline Base.length(x::BatchedArray) = length(x.data)
@inline Base.size(x::BatchedArray) = size(x.data)
@inline Base.size(x::BatchedArray, dim::Int) = size(x.data, dim)
Base.eltype(x::BatchedArray{T}) where T = T

batchsize(x::BatchedArray) = length(x.dims)
getdata(x::BatchedArray) = x.data

function Base.Array(x::BatchedArray)
    reshape(x.data, Base.front(size(x))..., x.dims[1], batchsize(x))
end

Base.similar(x::BatchedArray) = BatchedArray(similar(x.data), x.dims)
Base.similar(x::BatchedArray, dims::Tuple) = BatchedArray(similar(x.data,dims), x.dims)
Base.similar(x::BatchedArray, dims::Int...) = similar(x, dims)

Base.convert(::Type{Ptr{T}}, x::BatchedArray{T}) where T = Base.unsafe_convert(Ptr{T}, x.data)
Base.unsafe_convert(::Type{Ptr{T}}, x::BatchedArray{T}) where T = Base.unsafe_convert(Ptr{T}, x.data)

Base.zeros(x::BatchedArray) = BatchedArray(zeros(x.data), x.dims)
Base.ones(x::BatchedArray) = BatchedArray(ones(x.data), x.dims)
Base.copy(x::BatchedArray) = BatchedArray(copy(x.data), x.dims)

function Base.cat(dim::Int, xs::Vector{BatchedArray{T,N}}) where {T,N}
    y = cat(dim, map(x -> x.data, xs)...)
    if dim == N
        dims = cat(1, map(x -> x.dims, xs)...)
    else
        dims = xs[1].dims
    end
    BatchedArray(y, dims)
end
Base.cat(dim::Int, xs::BatchedArray...) = cat(dim, [xs...])

Base.fill!(x::BatchedArray, value) = fill!(x.data, value)
Base.sum(x::BatchedArray) = sum(x.data)

function Base.vec{T,N}(x::BatchedArray{T,N})
    s = stride(x.data, N)
    BatchedArray(vec(x.data), map(n -> n * s, x.dims))
end

+(x1::BatchedArray, x2::BatchedArray) = BatchedArray(x1.data + x2.data, x1.dims)
-(x1::BatchedArray, x2::BatchedArray) = BatchedArray(x1.data - x2.data, x1.dims)
*(x1::Array, x2::BatchedArray) = BatchedArray(x1 * x2.data, x2.dims)

Base.broadcast(::typeof(+), x1::Array, x2::BatchedArray) = BatchedArray(x1 .+ x2.data, x2.dims)
Base.broadcast(::typeof(+), x1::BatchedArray, x2::Array) = BatchedArray(x1.data .+ x2, x1.dims)
Base.broadcast(::typeof(+), x1::BatchedArray, x2::BatchedArray) = throw("Not implemented yet.")

function gemm(tA, tB, alpha, A::Array, B::BatchedArray)
    BatchedArray(gemm(tA, tB, alpha, A, B.data), B.dims)
end
