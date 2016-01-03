abstract AbstractVar{T,N}

data{T,N}(v::AbstractVar{T,N}) = v.value, v.grad

Base.size{T,N}(v::AbstractVar{T,N}, d::Int) = size(v.value, d)
Base.size{T,N}(v::AbstractVar{T,N}) = size(v.value)
Base.length{T,N}(v::AbstractVar{T,N}) = length(v.value)
Base.eltype{T,N}(v::AbstractVar{T,N}) = T
Base.ndims{T,N}(v::AbstractVar{T,N}) = N
