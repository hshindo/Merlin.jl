type Variable
  value::AFArray
  grad
  f
  args
end

Variable(value::Array) = Variable(AFArray(value))
Variable(value=nothing) = Variable(value, nothing, nothing, [], nothing)

function Base.call(f::Functor, args::Vector{Variable})
  v = Variable()
  v.f = f
  v.args = args
  forward!(f, v)
  v
end
Base.call(f::Functor, arg::Variable) = call(f, [arg])
Base.call(f::Functor, args::Variable...) = call(f, [args...])
function Base.call(fs::Vector, arg::Variable)
  for f in fs
    arg = call(f, arg)
  end
  arg
end

Base.getindex(v::Variable, key) = v.args[key]
Base.setindex!(v::Variable, value, key) = v.args[key] = value

function backward!(var::Variable)
  sorted = topsort(var)
  for i = length(sorted):-1:1
    v = sorted[i]
    length(v.args) == 0 && continue
    backward!(v.f, v)
  end
end

function addgrad!(var::Variable, grad)
  var.grad == nothing ? var.grad = grad : axpy!(1.0, grad, var.grad)
end

function topsort(var::Variable)
  sorted = Variable[]
  dict = ObjectIdDict()
  function visit(v::Variable)
    c = get!(dict, v, 1)
    if c == 1
      for a in v.args
        visit(a)
      end
      push!(sorted, v)
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
