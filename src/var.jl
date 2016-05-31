type Var
  value
  grad
  f
  args::Vector{Var}
end

Var(value; grad=false) = Var(value, grad ? zeros(value) : nothing, nothing, Var[])
Var() = Var(nothing)

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

hasgrad(v::Var) = v.grad != nothing

function forward(f, args::Vector{Var})
  any(a -> a.value == nothing, args) && return Var(nothing, nothing, f, args)
  y = Var(nothing, nothing, f, args)
  forward!(f, y)
  y
end

function topsort(var::Var)
  sorted = Var[]
  dict = ObjectIdDict()
  function visit(v::Var)
    if !haskey(dict, v)
      dict[v] = v
      for a in v.args
        visit(a)
      end
      push!(sorted, v)
    end
  end
  visit(var)
  sorted
end

function gradient!(var::Var)
  sorted = topsort(var)
  hasgrad(var) || (var.grad = ones(var.value))
  for i = 1:length(sorted)-1 # excludes top
    v = sorted[i]
    (hasgrad(v) || isempty(v.args)) && continue
    v.grad = zeros(v.value)
  end
  for i = length(sorted):-1:1
    v = sorted[i]
    v.f == nothing || backward!(v.f, v)
  end
  sorted
end
