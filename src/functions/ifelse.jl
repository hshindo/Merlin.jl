import Base.ifelse

ifelse(cond::Var, fx, xargs::Tuple, fy, yargs::Tuple) = forward0(ifelse, cond, fx, xargs, fy, yargs)

function forward(::typeof(ifelse), cond::Bool, fx, xargs, fy, yargs)
    cond ? fx(xargs...) : fy(yargs...)
end

export equals

equals(x::Var, y) = forward0(equals, x, y)
equals(x, y::Var) = forward0(equals, x, y)

forward(::typeof(equals), x, y) = x == y, nothing
