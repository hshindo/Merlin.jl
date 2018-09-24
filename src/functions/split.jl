doc"""
    split(x::Var, size::Vector)
    split(x::Var, dim::Int, size::Vector{Int})

# Example
```julia
T = Float32
x = Var(rand(T,10,10))
ys1 = split(x, [(5,10),(5,10)])
ys2 = split(x, 2, [2,3,5])
```
"""
function Base.split(x::Array{T,N}, size::Tuple) where {T,N}
    offset = 0
    map(size) do s
        p = pointer(x, offset+1)
        y = unsafe_wrap(typeof(x), p, s)
        y = Var(y, (split,x,offset))
        offset += length(y)
        y
    end
end

function Base.split(x::Var, dim::Int, size::Vector{Int})
    @assert sum(size) == Base.size(x,dim)
    if dim == ndims(x)
        offset = 0
        front = Base.front(Base.size(x))
        map(size) do s
            p = pointer(x.data, offset+1)
            y = unsafe_wrap(typeof(x.data), p, (front...,s))
            y = Var(y, (split,x,dim,offset))
            offset += length(y)
            y
        end
    else
        throw("Not implemented yet.")
    end
end
Base.split(x::Node, args...) = Node(split, x, args...)

function addgrad!(y::Var, ::typeof(split), x::Var, dim::Int, offset::Int)
    isvoid(x.grad) && return
    addto!(x.grad, offset, y.grad, 1, length(y))
end
