export Xavier

doc"""
    Xavier()

Xavier initialization.
"""
struct Xavier
end

function (::Xavier)(::Type{T}, dims::Int...) where T
    v = 2 / sum(dims)
    Normal(0,v)(T, dims...)
end
