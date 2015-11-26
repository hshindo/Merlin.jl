type Variable
  value
  work
  fun
  tails::Vector{Variable}
end

Variable(value) = Variable(value, nothing, nothing, Variable[])

function Base.|>(var::Variable, fun::Functor)
  y = apply(fun, var.value)
  typeof(y) <: Tuple ? Variable(y[1], y[2], fun, [var]) : Variable(y, nothing, fun, [var])
end

function Base.|>(vars::Vector{Variable}, fun::Functor)
  x = map(v -> v.value, vars)
  y = apply(fun, x)
  typeof(y) <: Tuple ? Variable(y[1], y[2], fun, vars) : Variable(y, nothing, fun, vars)
end

function diff!(var::Variable)
  var.grad = ones(var.data[1])
  sorted = topdown(var)
  for v in var
    length(v.tails) == 0 && continue
    if length(v.tails) == 1
      tail = v.tails[1]
      gradin = diff(v.fun, tail.data[1], v.grad)
      tail.grad == nothing ? tail.grad = gradin : tail.grad += gradin
    else
      inputs = map(t -> t.data, v.tails)
      gradins = diff(v.fun, inputs, v.grad)
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
