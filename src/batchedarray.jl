export BatchedArray
export batchsize
import Base: +, .+, -, .-, *, .*
import Base.LinAlg.BLAS: gemv, gemm

immutable BatchedArray{T,N}
    data::Array{T,N}
    size::Vector{Int}
end

typealias BatchedVector{T} BatchedArray{T,1}
typealias BatchedMatrix{T} BatchedArray{T,2}

function BatchedArray{T,N}(data::Vector{Array{T,N}})
    BatchedArray(cat(N, data...), map(x -> size(x,N), data))
end
BatchedArray(data::Array...) = BatchedArray([data...])

batchsize(x::BatchedArray) = length(x.size)

@inline Base.getindex(x::BatchedArray, key) = getindex(x.data, key)
@inline Base.setindex!(x::BatchedArray, value, key) = setindex!(x.data, value, key)

Base.length(x::BatchedArray) = length(x.data)
Base.size(x::BatchedArray) = size(x.data)
Base.size(x::BatchedArray, dim::Int) = size(x.data, dim)

Base.similar(x::BatchedArray) = BatchedArray(similar(x.data), x.size)
Base.similar(x::BatchedArray, dims::Tuple) = BatchedArray(similar(x.data,dims), x.size)

Base.convert{T}(::Type{Ptr{T}}, x::BatchedArray{T}) = Base.unsafe_convert(Ptr{T}, x.data)
Base.unsafe_convert{T}(::Type{Ptr{T}}, x::BatchedArray{T}) = Base.unsafe_convert(Ptr{T}, x.data)

function Base.vec{T,N}(x::BatchedArray{T,N})
    s = stride(x.data, N)
    BatchedArray(vec(x.data), map(n -> n * s, x.size))
end

function Base.split{T,N}(x::BatchedArray{T,N})
    n = stride(x.data, N)
    inds = Any[Colon() for i=1:N]
    cumsize = 0
    map(x.size) do s
        inds[N] = cumsize+1:cumsize+s
        cumsize += s
        view(x.data, inds...)
    end
end

.+(x1::Array, x2::BatchedArray) = BatchedArray(x1 .+ x2.data, x2.size)
.+(x1::BatchedArray, x2::Array) = BatchedArray(x1.data .+ x2, x1.size)
.+(x1::BatchedArray, x2::BatchedArray) = throw("Not implemented yet.")

*{T}(x1::Array{T}, x2::BatchedArray{T}) = gemm('N', 'N', T(1), x1, x2)

function gemm(tA, tB, alpha, A::Array, B::BatchedArray)
    BatchedArray(gemm(tA, tB, alpha, A, B.data), B.size)
end
