export ArrayVar

type ArrayVar{T,N} <: Var
    data::Array{T,N}
    grad::Array{T,N}
    capacity::Int
    f::Functor
    args::Vector{Var}
end

typealias MatrixVar{T} ArrayVar{T,2}
typealias VectorVar{T} ArrayVar{T,1}

ArrayVar(data, grad) = ArrayVar(data, grad, length(data), Identity(), Var[])
ArrayVar(data) = ArrayVar(data, zeros(data))
function ArrayVar(data::Array, f::Functor, args::Vector{Var})
    v = ArrayVar(data, typeof(data)(), length(data), f, args)
    forward!(f, v)
    v
end
ArrayVar(T::Type, dims::Tuple, f::Functor, args::Vector{Var}) = ArrayVar(Array(T,dims), f, args)

Base.isconst(v::ArrayVar) = isempty(v.grad)
Base.similar{T}(x::ArrayVar{T}, f::Functor, args::Vector{Var}) = ArrayVar(Array(T,size(x)), f, args)

function Base.resize!{T,N}(v::ArrayVar{T,N}, dims::Tuple{Vararg{Int,N}})
    count = prod(dims)
    @assert count > 0
    if count < v.capacity
        v.data = unsafe_wrap(Array, pointer(v.data), dims)
        v.grad = unsafe_wrap(Array, pointer(v.grad), dims)
    elseif count > v.capacity
        v.data = similar(v.data, dims)
        v.grad = Array{T,N}()
        v.capacity = count
    end
    v
end
Base.resize!(v::ArrayVar, dims::Int...) = resize!(v, dims)
