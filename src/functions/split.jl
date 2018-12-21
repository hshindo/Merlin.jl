import Base.split

function split(x::Var)
    @assert isa(x.data,Tuple)
    @assert isnothing(x.grad)
    i = 0
    ys = map(x.data) do d
        i += 1
        Var(d, ∇split!, (x,i))
    end
    x.grad = Array{Any}(nothing, i)
    ys
end

function split(x::Var, dims::Vector{Int})
    @assert isnothing(x.grad)
    ys = Var[]
    off = 0
    front = Base.front(size(x))
    for i = 1:length(dims)
        d = dims[i]
        ydata = x.data[off+1:off+d]
        ydata = reshape(ydata, front..., d)
        y = Var(ydata, ∇split!, (x,dims,i))
        push!(ys, y)
        off += d
    end
    x.grad = Array{Any}(nothing, length(dims))
    ys
end

function ∇split!(y::Var, x::Var, i::Int)
    x.grad[i] = y.grad
end

function ∇split!(y::Var, x::Var, cumdims::Vector{Int}, i::Int)
    isnothing(x.grad) && return
    
end

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
function split(x::Array{T,N}, size::Tuple) where {T,N}
    offset = 0
    map(size) do s
        p = pointer(x, offset+1)
        y = unsafe_wrap(typeof(x), p, s)
        y = Var(y, (split,x,offset))
        offset += length(y)
        y
    end
end

function split(x::Var, dim::Int, size::Vector{Int})
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
split(x::Node, args...) = Node(split, x, args...)

function addgrad!(y::Var, ::typeof(split), x::Var, dim::Int, offset::Int)
    isvoid(x.grad) && return
    addto!(x.grad, offset, y.grad, 1, length(y))
end
