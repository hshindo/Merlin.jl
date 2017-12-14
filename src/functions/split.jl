import Base.split
export unsafe_split

function split(x::Var, dims::Vector{Int})
    @assert sum(dims) == size(x)[end]
    front = Base.front(size(x))
    s = stride(x, ndims(x))
    cumdim = 0
    ys = Array{eltype(x),ndims(x)}[]
    for d in dims
        p = pointer(x, s*cumdim+1)
        y = unsafe_wrap(Array, p, (front...,d))
        push!(ys, y)
        cumdim += d
    end
    Var(ys, (split,x,dims))
end

#=
function unsafe_split(x::Array{T,N}, splitdims::Vector{Int}) where {T,N}
    sum(splitdims) == size(x,N) || throw("Invalid splitdims: $splitdims.")
    length(splitdims) == 1 && return [x]
    front = Base.front(size(x))
    m = prod(front)
    cumdim = 0
    ys = Array{T,N}[]
    for d in splitdims
        p = pointer(x, m*cumdim+1)
        y = unsafe_wrap(Array, p, (front...,d))
        push!(ys, y)
        cumdim += d
    end
    ys
end
=#

#=
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
=#
