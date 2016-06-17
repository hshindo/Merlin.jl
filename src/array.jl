Base.rand{T}(::Type{Array}, ::Type{T}, dims) = rand(T, dims)
Base.rand{T}(::Type{Array}, ::Type{T}, dims::Int...) = rand(T, dims)
Base.rand{T}(::Type{CuArray}, ::Type{T}, dims) = CuArray(rand(T,dims))
Base.rand{T}(::Type{CuArray}, ::Type{T}, dims::Int...) = CuArray(rand(T,dims))

Base.zeros(::Type{Array}, ::Type{T}, dims) = zeros(T, dims)
