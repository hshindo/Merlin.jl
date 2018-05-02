function add!(dest::AbstractArray{T}, src::AbstractArray{T}) where T
    @assert length(dest) == length(src)
    broadcast!(+, dest, dest, src)
end

function add!(dest::Array{T}, src::Array{T}) where T
    @assert length(dest) == length(src)
    BLAS.axpy!(T(1), src, dest)
    dest
end

add!(dest::AbstractCuArray, src::AbstractCuArray) = CUDA.add!(dest, src)
