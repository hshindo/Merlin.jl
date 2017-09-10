function init{T}(name::Symbol, ::Type{T}, dims::Int...)
    if name == :xavier
        randn(T,dims) * T(sqrt(2/))
    elseif name == :msr
    end
end

function init_xavier{T}(::Type{T}, dims::Tuple, insize::Int, outsize::Int)
    randn(T,dims) * T(sqrt(2/(insize+outsize)))
end
