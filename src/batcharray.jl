struct BatchedArray{T,N}
    data::Array{T,M}
    cumdims::Vector{Int}
end

BatchedArray(data::Array, dims::Int...) = BatchArray(data, [dims...])

batchdims(x::BatchArray) = x.dims
