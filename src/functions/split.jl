export unsafe_split

doc"""
    split(x::Var, dim::Int, batchdims::Vector{Int})

# Example
```julia
T = Float32
x = Var(rand(T,10,10))
ys = split(x, 2, [2,3,5])
```
"""
function Base.split(x::Var, dim::Int, dims::Vector{Int})
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

function unsafe_split(x::Var, dims::Vector{Int})
    ys = unsafe_split(x.data, dims)

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
