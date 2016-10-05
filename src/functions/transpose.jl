import Base.transpose

"""
    transpose(x::Var)
"""
@graph function transpose(x::Var)
    y = transpose(x.data)
    df(gy::UniArray) = isconst(x) || (x.grad .+= transpose(gy))
    Var(y, [x], transpose, df)
end
