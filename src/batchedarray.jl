export BatchedArray
export batchsize, isconstsize, unsafe_split
import Base: +, .+, -, .-, *, .*
import Base.LinAlg.BLAS: gemv, gemm

immutable BatchedArray{T,N}
    data::Array{T,N}
    dims::Vector{Int}
end

BatchedVector{T} = BatchedArray{T,1}
BatchedMatrix{T} = BatchedArray{T,2}

function BatchedArray{T,N}(arrays::Vector{Array{T,N}})
    data = cat(N, arrays...)
    dims = map(x -> size(x,N), arrays)
    BatchedArray(data, dims)
end
BatchedArray(data::Array...) = BatchedArray([data...])

@inline Base.getindex(x::BatchedArray, key) = getindex(x.data, key)
@inline Base.setindex!(x::BatchedArray, value, key) = setindex!(x.data, value, key)

Base.length(x::BatchedArray) = length(x.data)
Base.size(x::BatchedArray) = size(x.data)
Base.size(x::BatchedArray, dim::Int) = size(x.data, dim)

batchsize(x::BatchedArray) = length(x.dims)
isconstsize(x::BatchedArray) = all(d -> d == x.dims[1], x.dims)

Base.similar(x::BatchedArray) = BatchedArray(similar(x.data), x.dims)
Base.similar(x::BatchedArray, dims::Tuple) = BatchedArray(similar(x.data,dims), x.dims)
Base.similar(x::BatchedArray, dims::Int...) = similar(x, dims)

Base.convert{T}(::Type{Ptr{T}}, x::BatchedArray{T}) = Base.unsafe_convert(Ptr{T}, x.data)
Base.unsafe_convert{T}(::Type{Ptr{T}}, x::BatchedArray{T}) = Base.unsafe_convert(Ptr{T}, x.data)

function Base.vec{T,N}(x::BatchedArray{T,N})
    s = stride(x.data, N)
    BatchedArray(vec(x.data), map(n -> n * s, x.dims))
end

function unsafe_split(x::BatchedArray{T,N}) where {T,N}
    i = 1
    front = Base.front(size(x))
    map(x.dims) do d
        p = pointer(x.data, i)
        i += d
        dims = (front..., d)
        unsafe_wrap(Array, p, dims)
    end
end

Base.broadcast(::typeof(+), x1::Array, x2::BatchedArray) = BatchedArray(x1 .+ x2.data, x2.dims)
Base.broadcast(::typeof(+), x1::BatchedArray, x2::Array) = BatchedArray(x1.data .+ x2, x1.dims)
Base.broadcast(::typeof(+), x1::BatchedArray, x2::BatchedArray) = throw("Not implemented yet.")

*{T}(x1::Array{T}, x2::BatchedArray{T}) = gemm('N', 'N', T(1), x1, x2)

function gemm(tA, tB, alpha, A::Array, B::BatchedArray)
    BatchedArray(gemm(tA, tB, alpha, A, B.data), B.dims)
end
