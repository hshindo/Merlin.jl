export CPUBackend, CUDABackend

struct CPUBackend
end

(::CPUBackend)(x::Array) = x
(::CPUBackend)(x::CuArray) = Array(x)
(backend::CPUBackend)(x) = compile(x, backend)

struct CUDABackend
    device::Int
end

(::CUDABackend)(x::Array) = CuArray(x)
(::CUDABackend)(x::Array{Int}) = CuArray(Array{Cint}(x))
(::CUDABackend)(x::CuArray) = x
(backend::CUDABackend)(x) = compile(x, backend)

function compile(v::Var, backend)
    Var(backend(v.data), v.args, grad=backend(v.grad))
end
compile(x, backend) = x
