type Variable
  value
  grad
  f
  args
  backward
end

Variable(value=nothing, grad=nothing) = Variable(value, grad, nothing, [], nothing)

@compat function (f::Functor)(args::Vector{Variable})
  args[1].value == nothing && return Variable(nothing, nothing, f, args, nothing)
  xs = map(a -> a.value, args)
  y, backward = forward(f, xs)
  Variable(y, nothing, f, args, backward)
end

@compat function (f::Functor)(arg::Variable)
  arg.value == nothing && return Variable(nothing, nothing, f, [arg], nothing)
  y, backward = forward(f, arg.value)
  Variable(y, nothing, f, [arg], backward)
end

Base.getindex(v::Variable, key) = v.args[key]
Base.setindex!(v::Variable, value, key) = v.args[key] = value
Base.eltype(v::Variable) = eltype(v.value)

function gradient!(var::Variable)
  var.grad == nothing && (var.grad = ones(var.value))
  sorted = topsort(var)
  #for i = 1:length(sorted)-1 # excludes var
  #  v = sorted[i]
  #  length(v.args) > 0 && (v.grad = zeros(v.value))
  #end
  for i = length(sorted):-1:1
    v = sorted[i]
    length(v.args) == 0 && continue
    gxs = v.backward(v.grad)
    length(gxs) == 0 && continue
    @assert (length(v.args) == length(gxs))
    for i = 1:length(gxs)
      gx, a = gxs[i], v[i]
      a.grad == nothing ? a.grad = gx : a.grad += gx
    end
  end
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
