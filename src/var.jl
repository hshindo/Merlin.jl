abstract Var
export Var

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

macro Var3(name, fields...)
  exprs = Expr[]
  for f in fields
    push!(exprs, f.args[1])
  #  s, t = f.args[1], f.args[2]
  #  push!(exprs, :($s::$t))
  end
  body = Expr(:block, exprs...)
  q = quote
    type $name <: Var
      $body
    end
  end
  q
end

function Var(name::Symbol, fields::Expr...)
  body = Expr(:block, :(data::Any), :(grad::Any), :(tails::Vector), fields...)
  quote
    type $name <: Var
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
