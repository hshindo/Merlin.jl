export CPUBackend, CUDABackend

struct CPUBackend
end

(::CPUBackend)(x::Array) = x
(::CPUBackend)(x::CuArray) = Array(x)
(backend::CPUBackend)(x) = compile(x, backend)

struct CUDABackend
    dev::Int
end

(::CUDABackend)(x::Array) = CuArray(x)
(::CUDABackend)(x::CuArray) = x
(backend::CUDABackend)(x) = compile(x, backend)

compile(x::Void, backend) = nothing
function compile(v::Var, backend)
    grad = isvoid(v.grad) ? nothing : backend(v.grad)
    Var(backend(v.data), v.args, grad=grad)
end

function compile(x::Node, backend)
    f = backend(x.f)
    args = map(x.args) do arg
        isa(arg,Var) ? backend(arg) : arg
    end
    Node(f, args..., x.name)
end

function compile(g::Graph, backend)
    nodes = map(backend, g.nodes)
    Graph(nodes, g.inids, g.outid)
end
