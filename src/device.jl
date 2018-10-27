export CPUDevice, GPUDevice

abstract type Device end

function Device(dev::Int)
    if str == "cpu"
        CPUDevice()
    elseif startswith(str, "gpu:")
        id = parse(Int, str[5:end])
        GPUDevice(id)
    else
        throw("Unknown device: $str.")
    end
end

struct CPUDevice <: Device
end

struct GPUDevice <: Device
    id::Int
end

(::CPUDevice)(x::Array) = x
(::CPUDevice)(x::CuArray) = Array(x)
(::CPUDevice)(x::CuArray{Cint}) = Array{Int}(Array(x))
(::GPUDevice)(x::CuArray) = x
(::GPUDevice)(x::Array{Int}) = CuArray(Array{Cint}(x))
(::GPUDevice)(x::Array) = CuArray(x)
