export Vars

mutable struct Arrays{T,N}
    data::Vector{Array{T,N}}
end

function Base.vec(xs::Arrays)
    p = pointer(xs[1])
    n = sum(length, xs.data)
    unsafe_wrap(Array, p, (n,))
end

mutable struct Vars
    var::Var
    dims::Tuple
end

function Vars(xs::Vector{Var})
    y = concat(dim, xs...)
    dims = ntuple(ndims(y)) do i
        i == dim ? map(x -> size(x,dim), xs) : size(y,i)
    end
    Vars(y, dims)
end
function Vars(xs::Vector{Array{T,N}}) where {T,N}
    y = cat(dim, xs...)
    dims = ntuple(ndims(y)) do i
        i == dim ? map(x -> size(x,dim), xs) : size(y,i)
    end
    Vars(Var(y), dims)
end

Base.size(x::Vars) = x.dims
Base.size(x::Vars, i::Int) = i <= ndims(x) ? x.dims[i] : 1
Base.ndims(x::Vars) = length(x.dims)

function Var(x::Vars)
    for s in size(x)
        if isa(s, Int)

        elseif isa(s, Tuple)

        end
    end
end

function Base.vec(xs::Vars)
    xs.var
end
