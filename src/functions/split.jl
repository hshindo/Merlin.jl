import Base.split
export unsafe_split

function split(x::Var, dim::Int, size::Vector{Int})
    y = split(x.data, dim, size)
    Var(y, split, (x,))
end

function unsafe_split(x::Array{T,N}, dims::Vector{Int}) where {T,N}
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

function split(x::Array, dim::Int, size::Int)
    s = Base.size(x,dim) ÷ size
    s * size == Base.size(x,dim) || throw("Invalid size is specified.")
    split(x, dim, fill(s,size))
end
