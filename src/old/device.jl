export todevice, todevice!

todevice(x, dev::Int) = x

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

todevice(t::NTuple{N,Var}, dev::Int) where N = map(x -> todevice(x,dev), t)
todevice(t::NTuple{N,Functor}, dev::Int) where N = map(x -> todevice(x,dev), t)

function todevice(f::Functor, dev::Int)
    T = typeof(f)
    args = []
    for i = 1:length(fieldnames(T))
        v = getfield(f, i)
        push!(args, todevice(v,dev))
    end
    T(args...)
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
