export Backend, CPUBackend, CUDABackend, OpenCLBackend
import Base.convert

abstract type Backend end

struct CPUBackend <: Backend
end

convert(::CPUBackend, x::Array) = x
convert(::CPUBackend, x::CuArray) = Array(x)

struct CUDABackend <: Backend
    device::Int
end

convert(::CUDABackend, x::Array) = CuArray(x)
convert(::CUDABackend, x::CuArray) = x

struct OpenCLBackend <: Backend
    device::Int
end

function convert(backend::Backend, v::Var)
    Var(convert(backend,v.data), v.args, grad=convert(backend,v.grad))
end
convert(backend::Backend, x) = x
