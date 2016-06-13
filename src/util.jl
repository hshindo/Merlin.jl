function argmax(x, dim::Int)
  _, index = findmax(x, dim)
  ind2sub(size(x), vec(index))[dim]
end

function Base.rand{T,N}(::Type{T}, low::Float64, high::Float64, dims::NTuple{N,Int})
  # sqrt(6 / (dims[1]+dims[2]))
  a = rand(T, dims) * (high-low) + low
  convert(Array{T,N}, a)
end

Base.randn{T}(::Type{T}, dims...) = convert(Array{T}, randn(dims))

empty{T}(::Type{Array{T,1}}) = Array(T, 0)
empty{T}(::Type{Array{T,2}}) = Array(T, 0, 0)
empty{T}(::Type{Array{T,3}}) = Array(T, 0, 0, 0)
empty{T}(::Type{Array{T,4}}) = Array(T, 0, 0, 0, 0)
empty{T}(::Type{Array{T,5}}) = Array(T, 0, 0, 0, 0, 0)
empty{T}(::Type{Array{T,6}}) = Array(T, 0, 0, 0, 0, 0, 0)
