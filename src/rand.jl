"""
Saxe et al., Exact solutions to the nonlinear dynamics of learning in deep linear neural networks.
http://arxiv.org/abs/1312.6120
"""
function orthogonal{T}(::Type{T}, dim1::Int, dim2::Int; scale=1.1)
    a = randn(T, dim1, dim2)
    u, _, v = svd(a)
    q = size(u) == (dim1,dim2) ? u : v
    q * scale
end

function uniform{T}(::Type{T}, a, b, dims::Tuple)
    a < b || throw("Invalid interval: [$a: $b]")
    r = rand(T, dims)
    r .*= T(b - a)
    r .+= T(a)
    r
end
uniform{T}(::Type{T}, a, b, dims::Int...) = uniform(T, a, b, dims)
