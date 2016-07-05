function topsort(top)
  sorted = []
  dict = ObjectIdDict()
  function visit(node)
    haskey(dict, node) && return
    dict[node] = node
    for t in tails(node)
      visit(t)
    end
    push!(sorted, node)
  end
  visit(top)
  sorted
end

hasgrad(l::Layer) = l.gy != nothing
isleaf(l::Layer) = isempty(tails(l))

function gradient!(top::Layer)
  sorted = topsort(top)
  hasgrad(top) || (top.grad = ones(top.value))
  for i = 1:length(sorted)-1 # excludes top
    l = sorted[i]
    isleaf(l) && continue
    l.gy = zeros(l.y)
  end
  for i = length(sorted):-1:1
    l = sorted[i]
    backward!(l)
  end
  sorted
end

macro checkargs(f, args)
  quote
    if any(a -> typeof(a.y) == Symbol, $args)
      return Var(Symbol(), $f, $args)
    end
  end
end
