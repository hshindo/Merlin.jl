export BatchedArray

immutable BatchedArray{T,N}
    data::Array{T,N}
    size::Vector{Int}
end

function BatchedArray{T,N}(data::Vector{Array{T,N}})
    BatchedArray(cat(N, map(vec,data)...), map(x -> size(x,N), data))
end
BatchedArray(data::Array...) = BatchedArray([data...])

@inline Base.getindex(x::BatchedArray, key) = getindex(x.data, key)

Base.length(x::BatchedArray) = length(x.data)
Base.size(x::BatchedArray) = size(x.data)
Base.size(x::BatchedArray, dim::Int) = size(x.data, dim)

Base.similar(x::BatchedArray) = BatchedArray(similar(x.data), x.size)
Base.similar(x::BatchedArray, dims::Tuple) = BatchedArray(similar(x.data,dims), x.size)
