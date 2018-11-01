export todevice, todevice!

function todevice(x::Array{T}, dev::Int) where T
    if dev < 0
        x
    else
        CUDA.setdevice(dev)
        if T == Int
            CuArray(Array{Cint}(x))
        else
            CuArray(x)
        end
    end
end

function todevice(x::CuArray{T}, dev::Int) where T
    if dev < 0
        if T == Cint
            Array{Int}(Array(x))
        else
            Array(x)
        end
    elseif CUDA.getdevice() != dev
        throw("Not implemented yet.")
    end
end

function todevice(x::Var, dev::Int)
    data = todevice(x.data, dev)
    grad = todevice(x.grad, dev)
    x = Var(data)
    x.grad = grad
    x
end

function todevice!(x::Var, dev::Int)
    x.data = todevice(x.data, dev)
    isnothing(x.grad) || (x.grad = todevice(x.grad,dev))
    x
end

function todevice(g::Graph, dev::Int)
    nodes = map(g.nodes) do n
        args = map(n.args) do arg
            isa(arg,Var) ? todevice(arg,dev) : arg
        end
        Node(n.f, args, n.name)
    end
    Graph(nodes, g.inputids, g.outputids)
end

#=
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
=#
