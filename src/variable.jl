type Variable{T}
  value::T
  grad::T
  f
  args
  backward!
end

Variable{T}(value::T, grad::T) = Variable(value, grad, nothing, [], nothing)
Variable{T}(value::T) = Variable(value, similar(value,eltype(value),0))

@compat function (f::Functor)(args::Vector{Variable})
  isempty(args[1].value) && return Variable(args[1].value, args[1].value, f, args, nothing)
  xs = map(a -> a.value, args)
  y, backward! = forward(f, xs)
  Variable(y, similar(y, eltype(y), 0), f, args, backward!)
end

@compat function (f::Functor)(arg::Variable)
  isempty(arg.value) && return Variable(arg.value, arg.value, f, [arg], nothing)
  y, backward! = forward(f, arg.value)
  Variable(y, similar(y, eltype(y), 0), f, (arg,), backward!)
end

Base.getindex(v::Variable, key) = v.args[key]
Base.setindex!(v::Variable, value, key) = v.args[key] = value
Base.eltype(v::Variable) = eltype(v.value)

function gradient!(var::Variable)
  isempty(var.grad) && (var.grad = ones(var.value))
  sorted = topsort(var)
  for v in sorted
    (v == var || length(v.args) > 0) && continue
    v.grad = zeros(v.value)
  end
  for i = length(sorted):-1:1
    v = sorted[i]
    length(v.args) == 0 && continue
    if typeof(v.args) <: Tuple

    end
    gxs = map(a -> a.grad, v.args)
    v.backward!(gxs, v.grad)
  end
end

function topsort(var::Variable)
  sorted = Variable[]
  dict = ObjectIdDict()
  function visit(v::Variable)
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
