export Fill, Normal, Orthogonal, OrthoNormal, Uniform, Xavier

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
    OrthoNormal
"""
struct OrthoNormal
end

function (o::OrthoNormal)(::Type{T}, dim1::Int, dim2::Int) where T
    I = eye(dim2)
    lr = 0.1
    eps = 0.05 / (dim1+dim2)
    tries = 0
    while tries < 10
        Q = randn(dim1, dim2) / sqrt(dim2)
        for i = 1:100
            QTQmI = Q' * Q - I
            loss = sum(QTQmI .^ 2 / 2)
            Q2 = Q .^ 2
            a = abs.(Q2 .+ sum(Q2,1) .+ sum(Q2,2) - 1.0) + eps
            Q -= lr * Q * QTQmI ./ a
            if maximum(Q) > 1e6 || loss > 1e6 || isinf(loss)
                tries += 1
                lr /= 2.0
                break
            end
        end
        return Matrix{T}(Q)
    end
    #Q = xp.random.randn(input_size, output_size) / xp.sqrt(output_size)
    throw("Generation failed.")
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
