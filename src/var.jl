type Var{T,N} <: AbstractVar{T,N}
  value::Array{T,N}
  grad::Array{T,N}
  fixed::Bool
  buffer::Vector{T}
end

function call{T,N}(::Type{Var{T,N}})
  value = Array(T, ntuple(_ -> 0, N))
  Var(value, value, false, Array(T,0))
end

Var{T,N}(::Type{T}, dims::NTuple{N,Int}) = resize!(Var{T,N}(), dims)

function Var{T}(value::Array{T}, grad::Array{T}=T[])
  v = Var(T, size(value))
  copy!(v.value, value)
  length(grad) > 0 && copy!(v.grad, grad)
  v
end

default{T,N}(v::Var{T,N}) = Var{T,N}()
function default{T}(v::Var, ::Type{T}, N::Int)
  value = Array(T, ntuple(_ -> 0, N))
  Var(value, value, false, Array(T,0))
end

function Base.resize!{T,N}(v::Var{T,N}, dims::NTuple{N,Int})
  dims == size(v.value) && return v
  newlen = prod(dims)
  newlen == 0 && return v
  newlen < 0 && error("length error")
  if length(v.buffer) < newlen * 2
    n = length(v.buffer)
    n == 0 && (n += 1)
    while n < newlen * 2
      n *= 2
    end
    v.buffer = Array(T, n)
  end
  v.value = pointer_to_array(pointer(v.buffer), dims)
  v.grad = pointer_to_array(pointer(v.buffer,newlen+1), dims)
  v
end
Base.resize!{T,N}(v::Var{T,N}, dims::Int...) = resize!(v, dims)
