import Base.ifelse

"""
    ifelse(cond::Var, x::Tuple, y::Tuple)

If cond is true, apply x, otherwise apply y.

```julia
T = Float32
cond = Var(true)
f = Linear(T,10,7)
x = Var(rand(T,10,5))
y = ifelse(cond, f, (,x), identity (y,))
```
"""
function ifelse(cond::Var, x::Tuple, y::Tuple)
    isvoid(cond.data) && return Var(nothing,(ifelse,cond,x,y))
    isa(cond.data, Bool) || throw("Condition is not bool.")
    if cond.data
        args = ntuple(i -> x[i+1], length(x)-1)
        x[1](args...)
    else
        args = ntuple(i -> y[i+1], length(y)-1)
        y[1](args...)
    end
end

#=
macro ifelse(c, x, y)
    call2var(c)
    quote
        $(esc(c))
    end
end

function call2var(expr::Expr)
    expr.head == :call && unshift!(expr.args, tovar)
    for arg in expr.args
        isa(arg,Expr) && call2var(arg)
    end
end

tovar(args...) = Var(nothing, args)
=#
