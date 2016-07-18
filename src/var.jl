export @Var
abstract Var

function topsort(top::Var)
  sorted = Var[]
  dict = ObjectIdDict()
  function visit(v)
    haskey(dict, v) && return
    dict[v] = v
    for t in v.tails
      visit(t)
    end
    push!(sorted, v)
  end
  visit(top)
  sorted
end

macro Var(name, fields...)
  exprs = Expr[]
  for f in fields
    push!(exprs, f)
  end
  body = Expr(:block, exprs...)
  quote
    type $name <: Var
      data
      grad
      tails::Vector
      $body
    end
  end
end

Base.getindex(v::Var, key::Int) = v.tails[key]
Base.setindex!(v::Var, value, key::Int) = v.tails[key] = value

hasdata(v::Var) = v.data != nothing && !(typeof(v.data) <: Symbol)
hasgrad(v::Var) = v.grad != nothing

"""
    checkargs(expr)

Check arguments and decide eager or lazy evaluation..
"""
macro checkargs(f, args)
  quote
    if any(a -> typeof(a.value) == Symbol, $args)
      return Var(Symbol(), $f, $args)
    end
  end
end
