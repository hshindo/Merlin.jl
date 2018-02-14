struct BatchArray{T,N}
    data::Array{T,M}
    dims::Vector{Int}
end

BatchArray(data::Array, dims::Int...) = BatchArray(data, [dims...])

batchdims(x::BatchArray) = x.dims
