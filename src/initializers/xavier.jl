export Xavier

doc"""
    Xavier()

Xavier initialization.
"""
struct Xavier
end

function (::Xavier){T}(::Type{T}, dims::Int...)
    v = 2 / sum(dims)
    Normal(0,v)(T, dims...)
end
