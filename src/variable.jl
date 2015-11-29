type Variable
  value
  work
  fun
  tails::Vector{Variable}
  grad
end

Variable(value) = Variable(value, nothing, nothing, Variable[], nothing)

function Base.|>(var::Variable, fun::Functor)
  value, work = apply(fun, var.value)
  Variable(value, work, fun, [var], nothing)
end

function Base.|>(vars::Vector{Variable}, fun::Functor)
  inputs = map(v -> v.value, vars)
  value, work = apply(fun, inputs)
  Variable(value, work, fun, vars, nothing)
end

function Base.|>(var::Variable, funs::Vector{Functor})
  for fun in funs
    var = var |> fun
  end
  var
end

function diff!(var::Variable)
  var.grad = ones(var.value)
  sorted = topdown(var)
  for v in sorted
    length(v.tails) == 0 && continue
    if length(v.tails) == 1
      tail = v.tails[1]
      gradin = v.work == nothing ? diff(v.fun, tail.value, v.grad) : diff(v.fun, tail.value, v.work, v.grad)
      tail.grad == nothing ? tail.grad = gradin : tail.grad += gradin
    else
      inputs = map(t -> t.value, v.tails)
      gradins = v.work == nothing ? diff(v.fun, inputs, v.grad) : diff(v.fun, inputs, v.work, v.grad)
      for i = 1:length(inputs)
        gradin, tail = gradins[i], v.tails[i]
        tail.grad == nothing ? tail.grad = gradin : tail.grad += gradin
      end
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
