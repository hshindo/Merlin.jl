import Base.transpose

"""
    transpose(x::Var)
"""
function transpose(x::Var)
    y = transpose(x.data)
    df(gy::UniArray) = hasgrad(x) && (x.grad .+= transpose(gy))
    Var(y, [x], transpose, df)
end
