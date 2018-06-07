function add!(dest::AbstractArray{T}, src::AbstractArray{T}) where T
    @assert length(dest) == length(src)
    broadcast!(+, dest, dest, src)
end

function add!(n::Int, dest::Array{T}, doff::Int, src::Array{T}, soff::Int) where T
    BLAS.axpy!(n, T(1), pointer(src,soff), 1, pointer(dest,doff), 1)
    dest
end

function add!(dest::Array{T}, src::Array{T}) where T
    @assert length(dest) == length(src)
    add!(length(dest), dest, 1, src, 1)
end

add!(dest::AbstractCuArray, src::AbstractCuArray) = CUDA.add!(dest, src)
