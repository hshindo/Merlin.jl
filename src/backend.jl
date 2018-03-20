export CPUBackend, CUDABackend

# abstract type Backend end

struct CPUBackend
end

(::CPUBackend)(x::Array) = x
(::CPUBackend)(x::CuArray) = Array(x)
(::CPUBackend)(x::CuArray{Cint}) = Array{Int}(Array(x))
(::CPUBackend)(x) = x
# (backend::CPUBackend)(x) = setbackend(backend, x)

struct CUDABackend
    dev::Int
end
CUDABackend() = CUDABackend(0)

(::CUDABackend)(x::Array) = CuArray(x)
(::CUDABackend)(x::Array{Int}) = CuArray(Array{Cint}(x))
(::CUDABackend)(x::CuArray) = x
(::CUDABackend)(x) = x
# (backend::CUDABackend)(x) = setbackend(backend, x)


setbackend(backend, x::Any) = x
function setbackend(backend, x::Var)
    Var(backend(x.data), x.args, backend(x.grad))
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
    Graph(nodes, g.inids, g.outids)
end

function setbackend(backend::String)
    if startswith(backend, "cpu")
    elseif startswith(backend, "cuda")
    end
end

# iscpu() = true
# iscuda() = startswith(BACKEND, "cuda")

#=
function setbackend!(x::Var)
    x.data = CONFIG.backend(x.data)
    isvoid(x.grad) || (x.grad = CONFIG.backend(x.grad))
    x
end

function setbackend!(xs::Var...)
    for x in xs
        setbackend!(x)
    end
    xs
end

function settype!(::CuArray, x::Var)
    if !isa(x.data, CuArray)
        x.data = CuArray(x.data)
        isvoid(x.grad) || (x.grad = CuArray(x.grad))
    end
end
function settype!(::Array, x::Var)
    if !isa(x.data, Array)
        x.data = Array(x.data)
        isvoid(x.grad) || (x.grad = Array(x.grad))
    end
end
=#
