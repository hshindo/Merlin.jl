import Base.transpose

"""
    transpose(x::Var)
"""
function transpose(x::Var)
    y = transpose(x.data)
    df(gy) = isvoid(x.grad) || BLAS.axpy!(eltype(gy)(1), transpose(gy), x.grad)
    Var(y, df, (x,))
end
transpose(x::Var{Void}) = Var(Void(), transpose, (x,))
