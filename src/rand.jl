"""
Saxe et al., http://arxiv.org/abs/1312.6120
"""
function orthogonal{T}(::Type{T}, dims::Tuple; scale=1.1)
    length(dims) <= 1 && throw("dims must be 2 or more dimension.")
    dim1, dim2 = dims[1], prod(dims[2:end])
    a = randn(T,dim1,dim2)
    u, _, v = svd(a)
    q = size(u) == dims ? u : v
    reshape(q*scale, dims)
end
orthogonal{T}(::Type{T}, dims::Int...; scale=1.1) = orthogonal(T, dims, scale=scale)
