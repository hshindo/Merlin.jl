type Var
  value
  f
  args::Vector
  grad
end

Var(value, f=nothing, args=Var[], grad=nothing) = Var(value, f, args, grad)
param(value) = Var(value, nothing, Var[], zeros(value))

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

hasgrad(v::Var) = v.grad != nothing
isparam(v::Var) = isempty(v.args) && v.grad != nothing

"""
    checkargs(expr)

Check arguments and decide eager or lazy evaluation.
"""
macro checkargs(f, args)
  quote
    if any(a -> typeof(a.value) == Symbol, $args)
      return Var(Symbol(), $f, $args)
    end
  end
end
