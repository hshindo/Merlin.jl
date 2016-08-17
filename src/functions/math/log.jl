import Base.log

"""
    log
"""
function log(x::Var)
    y = log(x.data)
    #df(gy) = hasgrad(x) && log
    Var(y, [x], log, df)
end
