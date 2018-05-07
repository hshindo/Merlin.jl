export Uniform

doc"""
    Uniform(a, b)
    Uniform(b)

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

function (u::Uniform)(::Type{T}, dims::Int...) where T
    r = rand(T, dims)
    r .*= (u.b - u.a)
    r .-= u.a
    r
end
