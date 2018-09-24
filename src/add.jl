export addto!, broadcast_addto!

function addto!(dest::Array{T}, doffs::Int, src::Array{T}, soffs::Int, n::Int) where T
    axpy!(n, T(1), pointer(src,soffs), 1, pointer(dest,doffs), 1)
end

function addto!(dest::Array{T}, src::Array{T}) where T
    @assert length(dest) == length(src)
    axpy!(length(dest), T(1), src, 1, dest, 1)
end

function addto!(dest::Array{T}, I::Tuple, src::Array{T}) where T
    broadcast!(+, view(dest,I...), src)
end

function addto!(dest::Array{T}, src::Array{T}, I::Tuple) where T
    addto!(dest, src[I...])
end

function addto!(dest::CuArray{T}, src::CuArray{T}) where T
    @assert length(dest) == length(src)
    axpy!(length(dest), T(1), src, 1, dest, 1)
end

function addto!(dest::CuArray{T}, doffs::Int, src::CuArray{T}, soffs::Int, n::Int) where T
    axpy!(n, T(1), pointer(src,soffs), 1, pointer(dest,doffs), 1)
end

addto!(dest::CuArray, I::Tuple, src::CuArray) = CUDA.addto!(dest, I, src)
addto!(dest::CuArray, src::CuArray, I::Tuple) = CUDA.addto!(dest, CuDeviceArray(src,I))

broadcast_addto!(dest::Array{T}, src::Array{T}) where T = dest .+= src
broadcast_addto!(dest::CuArray, src::CuArray) = CUDA.broadcast_addto!(dest, src)
