type Variable
  value
  fun
  args::Tuple
end

function apply(fun::Functor, vars::Variable...)
  fun = clone(fun)
  inputs = map(v -> v.value, vars)
  output = apply(fun, inputs)
  Variable(output, fun, vars)
end

Base.|>(var::Variable, fun::Functor) = apply(fun, var)
Base.|>(vars::Tuple{Vararg{Variable}}, fun::Functor) = apply(fun, vars...)

function diff!(var::Variable, grad)
  var.grad = grad
  for v in topdown(var)
    length(v.tails) == 0 && continue
    inputs = map(t -> t.value, v.tails)
    gradins = diff(v.fun, v.grad, v.work, inputs...)
    for i = 1:length(gradins)
      g, t = gradins[i], v.tails[i]
      t.grad == nothing ? t.grad = g : t.grad += g
    end
  end
end

function topdown(var::Variable)
  sorted = Variable[]
  dict = ObjectIdDict()
  function visit(v::Variable)
    if !haskey(dict, v)
      push!(sorted, v)
      dict[v] = nothing
      for t in v.tails
        visit(t)
      end
    end
  end
  visit(var)
  sorted
end

function optimize!(opt::Optimizer, funs::Vector{Functor})
  for fun in funs
    applicable(optimize!, opt, fun) && optimize!(opt, fun)
  end
end
