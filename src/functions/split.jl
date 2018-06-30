doc"""
    split(x::Var, dim::Int, size::Vector{Int})

# Example
```julia
T = Float32
x = Var(rand(T,10,10))
ys = split(x, 2, [2,3,5])
```
"""
function Base.split(x::Var, dim::Int, size::Vector{Int})
    @assert sum(size) == Base.size(x,dim)
    if dim == ndims(x)
        cumdim = 0
        front = Base.front(Base.size(x))
        m = prod(front)
        ys = Var[]
        for s in size
            range = cumdim+1:cumdim+s
            data = view(x.data, front..., range)
            y = Var(data, (x,dim,range))
            push!(ys, y)
            cumdim += s
        end
    else
        throw("Not implemented yet.")
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
