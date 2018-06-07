export unsafe_split

doc"""
    split(x::Var, dim::Int, dims::Vector{Int})

# Example
```julia
T = Float32
x = Var(rand(T,10,10))
ys = split(x, 2, [2,3,5])
```
"""
function Base.split(x::Var, size::Vector)
    if dim == ndims(x)
        copy()
    else

    end

    cumdim = 0
    for d in dims
        y = x[]
        cumdim += d
    end
    ys

    @assert dim == ndims(x)
    @assert sum(dims) == size(x,dim)
    front = Base.front(size(x))
    cumdim = 0
    ys = Var[]
    for d in dims
        y = x[front...,cumdim+1:cumdim+d]
        push!(ys, y)
        cumdim += d
    end
    ys
end

function Base.split(x::Var, size::Vector)
    ys = Var[]
    offset = 1
    for s in size
        
        view(x.data, I...)
        p = pointer(x.data, offset)
        a = unsafe_wrap(Array, p)
        y = Var(a, (unsafe_split,x,size))
        push!(ys, y)
        offset += prod(s)
    end
    ys
end

function unsafe_split(x::UniArray{T,N}, dims::Vector{Int}) where {T,N}
    sum(dims) == size(x,N) || throw("Invalid dims: $dims.")
    length(dims) == 1 && return [x]

    cumdim = 0
    front = Base.front(size(x))
    m = prod(front)
    ys = typeof(x)[]
    for d in dims
        y = unsafe_array(x, m*cumdim+1, (front...,d))
        push!(ys, y)
        cumdim += d
    end
    ys
end
