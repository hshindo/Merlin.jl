export Vars

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
    for s in x.size
        if isa(s, Int)
        elseif isa(s, Vector{Int})

        end
    end
end

function Base.reshape(xs::Vars)

end
