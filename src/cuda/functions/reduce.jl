function argmax(x::CuArray, dim::Int)
    findmax(x,dim)[2]
end

function argmin(x::CuArray, dim::Int)
    findmin(x,dim)[2]
end

#=
function Base.maximum(x::CuArray, dim::Int)
    y, idx = CUDNN.findmax(x, dim)
    y
end

Base.findmax(x::CuArray, dim::Int) = CUDNN.findmax(x, dim)
argmax(x::CuArray, dim::Int) = CUDNN.findmax(x,dim)[2]

function sum(out, x::CuArray, dim::Int)
    y = CUDNN.sum(x, dim)
    isvoid(out) && return y
    out.data = y
    out.∇! = () -> begin

    end
end
=#
#=
function ∇maximum!(gy::CuDeviceArray, gx::CuDeviceArray, idx::CuDeviceArray)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    gx[idx[i]] += gy[i]
    return nothing
end
=#
