export Arrays

immutable Arrays{T,N}
    array::Array{T,N}
    dims::Vector{Int}
end

function Arrays{T,N}(data::Vector{Array{T,N}})
    Arrays(cat(N, map(vec,data)...), map(x -> size(x,N), data))
end

Base.length(x::Arrays) = length(x.array)
Base.size(x::Arrays) = size(x.array)
#Base.reshape(x::Arrays, dims::Tuple) = reshape(x.data, dims)
#Base.reshape(x::Arrays, dims::Int...) = reshape(x, dims)
