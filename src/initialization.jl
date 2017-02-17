function ortho{T}(::Type{T}, dim1, dim2)
    a = randn(T,dim1,dim2)
    u,_,v = svd(a)
    size(u) == size(a)

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return K.variable(scale * q[:shape[0], :shape[1]], name=name)
end
