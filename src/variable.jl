function alloc_cpu{T}(::Type{T}, dims)
  Array(T, dims)
end
alloc_cpu{T}(::Type{T}, dims...) = alloc_cpu(T, dims)

type VarBuffer
  buffer::Vector
end

type Variable
  value
  grad
  state
  f
  args
  b
end

const varbuf = VarBuffer([])

function Variable(a::Array, b::Bool)
  value = AFArray(a)
  v = Variable(value, nothing, nothing, nothing, [], b)
  #b ? finalizer(value, release) : push!(varbuf.buffer, v)
  v
end

#Variable(value=nothing, grad=nothing) = Variable(value, grad, nothing, nothing, [])

function reset2()
  while length(varbuf.buffer) > 0
    v = pop!(varbuf.buffer)
    release(v.value)
  end
end

function reset()
  for v in varbuf.buffer
    v.b || finalize(v.value)
    #release(v.value)
  end
  varbuf.buffer = []
end

function call(f::Functor, args::Vector{Variable})
  y = Variable(nothing, nothing, nothing, f, args, false)
  # y.f = f
  # y.args = args
  forward!(f, y)
  push!(varbuf.buffer, y)
  y
end
call(f::Functor, arg::Variable) = call(f, [arg])

function call(fs::Vector, arg::Variable)
  for f in fs
    arg = call(f, arg)
  end
  arg
end

getindex(v::Variable, key) = v.args[key]
setindex!(v::Variable, value, key) = v.args[key] = value

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
