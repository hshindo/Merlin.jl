export argmax

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

function softmax{T}(x::Matrix{T})
  y = similar(x)
  max = maximum(x, 1)
  for j = 1:size(x,2)
    z = T(0)
    @inbounds @simd for i = 1:size(x,1)
      z += exp(x[i,j] - max[j])
    end
    z == T(0) && error("z == 0")
    @inbounds @simd for i = 1:size(x,1)
      y[i,j] = exp(x[i,j] - max[j]) / z
    end
  end
  y
end

function logsoftmax{T}(x::Matrix{T})
  y = similar(x)
  max = maximum(x, 1)
  for j = 1:size(x,2)
    sum = T(0)
    @inbounds @simd for i = 1:size(x,1)
      sum += exp(x[i,j] - max[j])
    end
    logz = log(sum)
    @inbounds @simd for i = 1:size(x,1)
      y[i,j] = x[i,j] - max[j] - logz
    end
  end
  y
end
