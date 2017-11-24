export Normal

doc"""
    Normal(mean, var)

Generator of ndarray with a normal distribution.

# Arguments
* mean: Mean of the random values.
* var: Variance of the random values.
"""
struct Normal
    mean
    var

    function Normal(mean, var)
        @assert var >= 0
        new(mean, var)
    end
end

function (n::Normal)(::Type{T}, dims::Int...) where T
    r = randn(T, dims)
    r .*= sqrt(n.var)
    r .+= n.mean
    r
end
