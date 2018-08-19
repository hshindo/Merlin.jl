export addto!
import .CUDA: addto!

function addto!(dest::AbstractArray{T}, src::AbstractArray{T}) where T
    @assert length(dest) == length(src)
    broadcast!(+, dest, dest, src)
end

function addto!(dest::Array{T}, doffs::Int, src::Array{T}, soffs::Int, n::Int) where T
    axpy!(n, T(1), pointer(src,soffs), 1, pointer(dest,doffs), 1)
    dest
end

function addto!(dest::Array{T}, src::Array{T}) where T
    @assert length(dest) == length(src)
    addto!(dest, 1, src, 1, length(dest))
end

# addto!(dest::AbstractCuArray, src::AbstractCuArray) = CUDA.addto!(dest, src)
