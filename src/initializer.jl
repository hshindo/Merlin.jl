export Fill, Normal, Orthogonal, Uniform, Xavier

doc"""
    Fill(x)

Fill initializer.
"""
struct Fill
    x
end

(f::Fill)(::Type{T}, dims::Int...) where T = fill(T(f.x), dims)

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

doc"""
    Orthogonal([gain=1.0])

# References
* Saxe et al., [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](http://arxiv.org/abs/1312.6120)
"""
struct Orthogonal
    gain::Float64
end
Orthogonal() = Orthogonal(1.0)

function (o::Orthogonal)(::Type{T}, dim1::Int, dim2::Int) where T
    a = randn(T, dim1, dim2)
    u, _, v = svd(a)
    q = size(u) == (dim1,dim2) ? u : v'
    q * T(o.gain)
end

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
