export CPUBackend, CUDABackend

struct CPUBackend
end

(::CPUBackend)(x::Array) = x
(::CPUBackend)(x::CuArray) = Array(x)
(backend::CPUBackend)(x) = setbackend(backend, x)

struct CUDABackend
    dev::Int
end

(::CUDABackend)(x::Array) = CuArray(x)
(::CUDABackend)(x::CuArray) = x
(backend::CUDABackend)(x) = setbackend(backend, x)

setbackend(backend, x::Any) = x
function setbackend(backend, x::Var)
    Var(backend(x.data), x.args, grad=backend(x.grad))
end
function setbackend(backend, x::Node)
    f = backend(x.f)
    args = map(x.args) do arg
        isa(arg,Var) ? backend(arg) : arg
    end
    Node(f, args..., name=x.name)
end
function setbackend(backend, g::Graph)
    nodes = map(backend, g.nodes)
    Graph(nodes, g.inids, g.outid)
end
