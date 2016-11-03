export ArrayVar

abstract Var

Base.eltype(v::Var) = eltype(v.data)
Base.size(v::Var) = size(v.data)
Base.size(v::Var, d::Int) = size(v.data, d)
Base.getindex(v::Var, key::Int) = v.args[key]
Base.setindex!(v::Var, value, key::Int) = v.args[key] = value

type Identity <: Functor
end

type ArrayVar{T,N} <: Var
    data::Array{T,N}
    grad::Array{T,N}
    capacity::Int
    f::Functor
    args::Vector{Var}
end

ArrayVar(data, grad) = ArrayVar(data, grad, length(data), Identity(), Var[])
ArrayVar(data) = ArrayVar(data, zeros(data))
constvar(data) = ArrayVar(data, typeof(data)())

function ArrayVar(data, f::Functor, args::Vector{Var})
    v = ArrayVar(data, typeof(data)(), length(data), f, args)
    forward!(f, v)
    v
end
ArrayVar(f::Functor, args::Var...) = ArrayVar(f, Var[args...])

function Base.resize!(v::ArrayVar, dims::Tuple{Vararg{Int}})
    count = prod(dims)
    @assert count > 0
    if count < v.capacity
        v.data = unsafe_wrap(Array, pointer(v.data), dims)
        v.grad = unsafe_wrap(Array, pointer(v.grad), dims)
    elseif count > v.capacity
        v.data = similar(v.data, dims)
        v.grad = similar(v.grad, dims)
        v.capacity = count
    end
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
