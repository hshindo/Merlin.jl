const HW_HANDLE = Libdl.dlsym(Native.library2, :getName)

function callhw()
  ccall(HW_HANDLE, Void,
    (Cint,), 1)
end

function alloc_cpu{T}(::Type{T}, dims)
  Array(T, dims)
end
alloc_cpu{T}(::Type{T}, dims...) = alloc_cpu(T, dims)

type Variable
  value
  grad
  state
  f
  args
end

Variable(value=nothing, grad=nothing) = Variable(value, grad, nothing, nothing, [])

function Base.call(f::Functor, args::Vector{Variable})
  y = Variable()
  y.f = f
  y.args = args
  forward!(f, y)
  y
end
Base.call(f::Functor, arg::Variable) = call(f, [arg])

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
