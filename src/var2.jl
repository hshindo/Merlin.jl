type Var{T}
  value::T
  grad::T
  fixed::Bool
  f
  args
  buffer::T
  work
end

Var(value) = Var(value, [], nothing, (), nothing)

constant(value) = Var(value, nothing, nothing, (), nothing)

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

function call(f::Functor, args)
  #any(a -> a.value == nothing, args) && return Var(nothing, [], f, args, nothing)
  x = map(a -> a.value, args)
  y, backward! = forward(f, x)
  Var(y, [], f, args, backward!)
end
call(f::Functor, args::Var...) = call(f, args)

function call(funs::Vector, arg::Var)
  for f in funs
    arg = f(arg)
  end
  arg
end

forward(f::Functor, args::Tuple) = forward(f, args...)

function backward!(var::Var)
  sorted = topdown(var)
  for v in sorted
    v.grad == nothing && continue
    length(v.args) == 0 && continue
    gx = map(v.args) do a
      a.grad == [] && (a.grad = zeros(a.value))
      a.grad
    end
    typeof(gx) <: Tuple ? v.backward!(v.grad, gx...) : v.backward!(v.grad, gx)
  end
end

function topdown(var::Var)
  sorted = Var[]
  dict = ObjectIdDict()
  function visit(v::Var)
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
