type Variable
  data
  work
  fun
  tails::Vector{Variable}
end

Variable(data, work) = Variable(data, work, nothing, Variable[])
Variable(data) = Variable(data, nothing)

function Base.|>(var::Variable, fun::Functor)
  h = apply(fun, var)
  h.fun = fun
  h.tails = [var]
  h
end

function Base.|>(vars::Vector{Variable}, fun::Functor)
  h = apply(fun, vars)
  h.fun = fun
  h.tails = vars
  h
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
