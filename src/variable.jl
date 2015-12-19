type Variable
  value
  grad
  args
  diff
  fixed::Bool
end

Variable(value, grad) = Variable(value, grad, (), [], false)
Variable(value) = Variable(value, nothing)

function call(f::Functor, arg::Variable)
  y, diff = apply(f, arg.value)
  Variable(y, nothing, (arg,), diff, false)
end

function call(f::Functor, args::Vector{Variable})
  x = map(a -> a.value, args)
  y, diff = apply(f, x)
  Variable(y, nothing, args, diff, false)
end

function call(f::Functor, args::Tuple{Vararg{Variable}})
  x = map(a -> a.value, args)
  y, diff = apply(f, x...)
  Variable(y, nothing, args, diff, false)
end

function call(funs::Vector, arg::Variable)
  for f in funs
    arg = f(arg)
  end
  arg
end

function diff!(var::Variable, grad)
  var.grad = grad
  sorted = topdown(var)
  for v in sorted
    length(v.args) == 0 && continue
    if typeof(v.args) <: Tuple
      length(v.args) == 1 || error("NOT IMPLEMENTED YET")
      arg, g = v.args[1], v.diff(v.grad)
      arg.grad == nothing ? arg.grad = g : arg.grad += g
    elseif typeof(v.args) <: Vector
      gs = v.diff(v.grad)
      for i = 1:length(v.args)
        arg, g = v.args[i], gs[i]
        arg.grad == nothing ? arg.grad = g : arg.grad += g
      end
    else
      error("")
    end
  end
end

function topdown(var::Variable)
  sorted = Variable[]
  dict = ObjectIdDict()
  function visit(v::Variable)
    c = get!(dict, v, 1)
    if c == 1
      push!(sorted, v)
      for a in v.args
        visit(a)
      end
    #else
    #  dict[v] = c + 1
    end
  end
  visit(var)
  sorted
end

function optimize!(opt::Optimizer, funs::Vector)
  for fun in funs
    applicable(optimize!, opt, fun) && optimize!(opt, fun)
  end
end
