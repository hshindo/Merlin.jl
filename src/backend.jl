export Backend, CPUBackend, CUDABackend, OpenCLBackend
export compile

abstract type Backend end

struct CPUBackend <: Backend
end

compile(x::Array, ::CPUBackend) = x
compile(x::CuArray, ::CPUBackend) = Array(x)

struct CUDABackend <: Backend
    device::Int
end

compile(x::Array, ::CUDABackend) = CuArray(x)
compile(x::Array{Int}, ::CUDABackend) = CuArray(Array{Cint}(x))
compile(x::CuArray, ::CUDABackend) = x

struct OpenCLBackend <: Backend
    device::Int
end

function compile(v::Var, backend::Backend)
    Var(compile(v.data,backend), v.args, grad=compile(v.grad,backend))
end
compile(x, backend::Backend) = x
