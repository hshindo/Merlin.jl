export Var

type Var
    data::Array
    grad::Array
    capacity::Int
    f::Functor
    args::Vector
end

Var(data::Array) = Var(data, zeros(data))
Var(data::Array, grad::Array) = Var(data, grad, length(data), NullFunctor(), Var[])

constant(data::Array) = Var(data, typeof(data)())

Base.getindex(v::Var, key::Int) = v.args[key]
Base.setindex!(v::Var, value, key::Int) = v.args[key] = value
Base.size(v::Var) = size(v.data)
Base.size(v::Var, d::Int) = size(v.data, d)

Base.isconst(v::Var) = isempty(v.grad)

function forward(f::Functor, v::Var)
    data = typeof(v.data)()
    grad = typeof(v.data)()
    o = Var(data, grad, length(data), f, [v])
    forward!(f, o)
    o
end
#forward(f::Functor, args::Var...) = forward(f, [args...])

function Base.resize!(v::Var, dims::Tuple{Vararg{Int}})
    len = prod(dims)
    @assert len > 0
    if len < v.capacity
        v.data = unsafe_wrap(Array, pointer(v.data), dims)
        v.grad = unsafe_wrap(Array, pointer(v.grad), dims)
    elseif len > v.capacity
        v.data = similar(v.data, dims)
        v.grad = similar(v.grad, dims)
    end
    v.capacity = len
    v
end

function topsort(top::Var)
    sorted = Var[]
    dict = ObjectIdDict()
    function visit(v::Var)
        haskey(dict,v) && return
        dict[v] = v
        for a in v.args
            visit(a)
        end
        push!(sorted, v)
    end
    visit(top)
    sorted
end

function gradient!(top::Var)
    sorted = topsort(top)
    isconst(top) && (top.grad = ones(top.data))
    for i = 1:length(sorted)-1 # excludes top
        v = sorted[i]
        isconst(v) || continue
        (!isconst(v) || isempty(v.args)) && continue
        fill!(v.grad, 0)
        #v.grad = zeros(v.data)
    end
    for i = length(sorted):-1:1
        v = sorted[i]
        backward!(v.f, v)
        #v.df == nothing || v.df(v.grad)
    end
    sorted
end
