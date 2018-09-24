import Base.transpose

"""
    transpose(x)
"""
function transpose(x::Var)
    y = transpose(x.data)
    Var(y, (transpose,x))
end

function addgrad!(y::Var, ::typeof(transpose), x::Var)
    isvoid(x.grad) && return
    addto!(transpose(y.grad), x.grad)
end
