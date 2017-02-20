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
