mutable struct Vars
    var::Var
    size::Tuple
end

Base.size(x::Vars) = x.size
Base.size(x::Vars, i::Int) = i <= length(x.size) ? x.size[i] : 1
Base.ndims(x::Vars) = length(x.size)

function Var(x::Vars)
    for s in x.size
        if isa(s, Int)
        elseif isa(s, Vector{Int})

        end
    end
end

function Base.reshape(xs::Vars)

end
