type Var
  value
  f
  args
  backward!
  grad
end

function Var(value, f=nothing, args=[], (backward!)=nothing)
  Var(value, f, args, backward!, nothing)
end
Var() = Var(nothing)
#Var(value; grad=true) = Var(value, nothing, [], nothing, zeros(value))

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

hasgrad(v::Var) = v.grad != nothing

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
    hasgrad(v) && continue
    length(v.args) == 0 && continue
    v.grad = zeros(v.value)
  end
  for i = length(sorted):-1:1
    v = sorted[i]
    v.backward! == nothing || v.backward!(v)
  end
  sorted
end
