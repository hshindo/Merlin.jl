function call(dep::Layer, head::Layer)
  forward(head, dep)
end

function backward(var::Variable)

end

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
