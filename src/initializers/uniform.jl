export Uniform

doc"""
    Uniform(a, b)

Generator of ndarray with a uniform distribution.

# Arguments
* a: Lower bound of the range of random values.
* b: Upper bound of the range of random values.
"""
struct Uniform
    a
    b

    function Uniform(a, b)
        @assert a <= b
        new(a, b)
    end
end

Uniform(b) = Uniform(-b, b)

function sample{T}(u::Uniform, ::Type{T}, dims::Int...)
    r = rand(T, dims)
    r .*= (u.b - u.a)
    r .-= u.a
    r
end
