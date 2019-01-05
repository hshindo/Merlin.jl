export todevice, todevice!

todevice(x) = x

function todevice(x::Array{T}) where T
    dev = getdevice()
    if dev < 0
        x
    else
        if T == Int
            CuArray(Array{Cint}(x))
        else
            CuArray(x)
        end
    end
end

function todevice(x::CuArray{T}) where T
    dev = getdevice()
    if dev < 0
        if T == Cint
            Array{Int}(Array(x))
        else
            Array(x)
        end
    elseif getdevice() != CUDA.getdevice(x)
        throw("Not implemented yet.")
    end
end

function todevice(x::Var)
    data = todevice(x.data)
    grad = todevice(x.grad)
    x = Var(data)
    x.grad = grad
    x
end

function todevice!(x::Var)
    x.data = todevice(x.data)
    x.grad = todevice(x.grad)
    x
end

function todevice(f::Functor)
    T = typeof(f)
    args = []
    for i = 1:length(fieldnames(T))
        v = getfield(f, i)
        push!(args, todevice(v))
    end
    T(args...)
end

todevice(t::NTuple{N,Var}) where N = map(todevice, t)
todevice(t::NTuple{N,Functor}) where N = map(todevice, t)

function todevice(g::Graph)
    throw("Not implemented yet.")
    nodes = map(g.nodes) do n
        args = map(n.args) do arg
            isa(arg,Var) ? todevice(arg,dev) : arg
        end
        Node(n.f, args, n.name)
    end
    Graph(nodes, g.inputids, g.outputids)
end
