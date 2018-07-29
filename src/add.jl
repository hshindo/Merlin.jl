import .CUDA: add!

function add!(dest::AbstractArray{T}, src::AbstractArray{T}) where T
    @assert length(dest) == length(src)
    broadcast!(+, dest, dest, src)
end

function add!(dest::Array{T}, doffs::Int, src::Array{T}, soffs::Int, n::Int) where T
    BLAS.axpy!(n, T(1), pointer(src,soffs), 1, pointer(dest,doffs), 1)
    dest
end

function add!(dest::Array{T}, src::Array{T}) where T
    @assert length(dest) == length(src)
    add!(dest, 1, src, 1, length(dest))
end

# add!(dest::AbstractCuArray, src::AbstractCuArray) = CUDA.add!(dest, src)
