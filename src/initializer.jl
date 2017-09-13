export Uniform, Normal, Xavier, Orthogonal

doc"""
    Uniform(a, b)

Initializer that generates tensors with a uniform distribution.

# Arguments
* a: Lower bound of the range of random values.
* b: Upper bound of the range of random values.
"""
struct Uniform
    a
    b
end

function random{T}(u::Uniform, ::Type{T}, dims::Int...)
    r = rand(T, dims)
    r .*= (u.b - u.a)
    r .-= u.a
    r
end

doc"""
    Normal(mean, var)

Initializer that generates tensors with a normal distribution.

# Arguments
* mean: Mean of the random values.
* var: Variance of the random values.
"""
struct Normal
    mean
    var
end

function random{T}(n::Normal, ::Type{T}, dims::Int...)
    r = randn(T, dims)
    r .*= sqrt(n.var)
    r .+= n.mean
    r
end

doc"""
    Xavier()

Xavier initialization.
"""
struct Xavier
end

function random{T}(x::Xavier, ::Type{T}, dims::Int...)
    v = 2 / sum(dims)
    random(Normal(0,v), T, dims...)
end

doc"""
    Orthogonal()

# References
* Saxe et al., [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](http://arxiv.org/abs/1312.6120)
"""
struct Orthogonal
end

function random{T}(o::Orthogonal, ::Type{T}, dim1::Int, dim2::Int)
    a = randn(T, dim1, dim2)
    u, _, v = svd(a)
    q = size(u) == (dim1,dim2) ? u : v
    q * T(scale)
end
