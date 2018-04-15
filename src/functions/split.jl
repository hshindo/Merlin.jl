import Base.split

doc"""
    split(x::Var, dims::Vector{Int})

# Example
```julia
T = Float32
x = Var(rand(T,10,10))
ys = split(x, [2,3,5])
```
"""
function split(x::Var, dim::Int, dims::Vector{Int})
    @assert sum(dims) == size(x,ndims(x))
    front = [Colon() for _=1:ndims(x)-1]
    cumdim = 0
    ys = Var[]
    for d in dims
        y = x[front...,cumdim+1:cumdim+d]
        push!(ys, y)
        cumdim += d
    end
    ys
end

function split(x::Array{T,N}, dim::Int, splitdims::Vector{Int}) where {T,N}
    sum(splitdims) == size(x,dim) || throw("Invalid splitdims.")
    cumdim = 0
    map(splitdims) do d
        range = ntuple(N) do i
            i == dim ? (cumdim+1:cumdim+d) : Colon()
        end
        cumdim += d
        view(x, range...)
    end
end

function unsafe_array(x::Array, index::Int, dims)
    p = pointer(x, index)
    unsafe_wrap(Array, p, dims)
end

function unsafe_array(x::CuArray, index::Int, dims)
    p = pointer(x, index)
    # mb = MemBlock(p, )
end

function unsafe_split(x::UniArray{T,N}, dims::Vector{Int}) where {T,N}
    sum(dims) == size(x,N) || throw("Invalid splitdims: $dims.")
    length(dims) == 1 && return [x]
    front = Base.front(size(x))
    m = prod(front)
    cumdim = 0
    ys = Array{T,N}[]
    for d in dims
        p = pointer(x, m*cumdim+1)
        y = unsafe_wrap(Array, p, (front...,d))
        push!(ys, y)
        cumdim += d
    end
    ys
end
