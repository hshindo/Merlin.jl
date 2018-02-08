export CPUBackend, CUDABackend

struct CPUBackend
end

(::CPUBackend)(x::Array) = x
(::CPUBackend)(x::CuArray{Cint}) = Array{Int}(Array(x))
(::CPUBackend)(x::CuArray) = Array(x)
(backend::CPUBackend)(x) = compile(x, backend)

function setbackend!(x::Var, backend::CPUBackend)
    backend(x.data)
    compile(f)
end

struct CUDABackend
    device::Int
end

(::CUDABackend)(x::Array) = CuArray(x)
(::CUDABackend)(x::Array{Int}) = CuArray(Array{Cint}(x))
(::CUDABackend)(x::CuArray) = x
(backend::CUDABackend)(x) = compile(x, backend)

compile(x, backend) = x
function compile(v::Var, backend)
    Var(backend(v.data), v.args, grad=backend(v.grad))
end

function compile(x::Node, backend)
    f = backend(x.f)
    args = map(backend, x.args)
    Node(f, args..., x.name)
end

function compile(g::Graph, backend)
    nodes = map(backend, g.nodes)
    Graph(nodes, g.inids, g.outid)
end
