type Variable
  value
  grad
  f
  args::Vector{Variable}
  backward!
end

function Variable(value=nothing, grad=nothing, f=nothing, args=Variable[])
  Variable(value, grad, f, args, nothing)
end

function forward(f::Functor, args::Vector{Variable})
  v = Variable(nothing, nothing, f, args)
  all(a -> a.value != nothing, args) && forward!(f, v)
  v
end
forward(f::Functor, arg::Variable) = forward(f, [arg])
forward(f::Functor, args::Variable...) = forward(f, [args...])

Base.getindex(v::Variable, key) = v.args[key]
Base.setindex!(v::Variable, value, key) = v.args[key] = value
Base.eltype(v::Variable) = eltype(v.value)

function gradient!(var::Variable)
  var.grad == nothing && (var.grad = ones(var.value))
  sorted = topsort(var)
  for v in sorted
    v == var && continue
    length(v.args) == 0 && continue
    v.grad = zeros(v.value)
  end
  for i = length(sorted):-1:1
    v = sorted[i]
    length(v.args) == 0 && continue
    v.backward!()
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

function gradcheck()

end
