export Xavier

doc"""
    Xavier()

Xavier initialization.
"""
struct Xavier
end

function sample{T}(x::Xavier, ::Type{T}, dims::Int...)
    v = 2 / sum(dims)
    sample(Normal(0,v), T, dims...)
end
