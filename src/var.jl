abstract Var
export Var

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
