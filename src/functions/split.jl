export unsafe_split

doc"""
    split(x::Var, dims::Vector{Int})

# Example
```julia
T = Float32
x = Var(rand(T,10,10))
ys = split(x, [2,3,5])
```
"""
function Base.split(x::Var, dim::Int, dims::Vector{Int})
    @assert sum(dims) == size(x,dim)
    front = [Colon() for _=1:ndims(x)-1]
    cumdim = 0
    ys = Var[]
    for d in dims
        a = ntuple(N) do i
            i == dim ? (cumdim+1:cumdim+d) : Colon()
        end
        y = x[front...,cumdim+1:cumdim+d]
        push!(ys, y)
        cumdim += d
    end
    ys
end

function unsafe_split(x::Array{T,N}, dims::Vector{Int}) where {T,N}
    sum(dims) == size(x,N) || throw("Invalid dims: $dims.")
    length(dims) == 1 && return [x]

    cumdim = 0
    front = Base.front(size(x))
    m = prod(front)
    ys = Array{T,N}[]
    for d in dims
        y = unsafe_wrap(Array, pointer(x,m*cumdim+1), (front...,d))
        push!(ys, y)
        cumdim += d
    end
    ys
end

function unsafe_split(x::CuArray{T,N}, dims::Vector{Int}) where {T,N}
    sum(dims) == size(x,N) || throw("Invalid dims: $dims.")
    length(dims) == 1 && return [x]

    cumdim = 0
    front = Base.front(size(x))
    m = prod(front)
    ys = CuArray{T,N}[]
    for d in dims
        y = unsafe_wrap(CuArray, pointer(x,m*cumdim+1), (front...,d))
        push!(ys, y)
        cumdim += d
    end
    ys
end
