type Node
  f::Functor
  tails::Vector{Node}

  function Node(f::Functor, tails::Vector{Node})
    if length(tails) == 0
      new(f, tails)
    elseif length(tails) == 1
      f(tails[1].f.y)
      new(f, tails)
    else
      x = map(t -> t.f.y, tails)
      f(x)
      new(f, tails)
    end
  end
end

Node(f::Functor) = Node(f, Node[])
Node(f::Functor, tail::Node) = Node(f, [tail])

function backward!(node::Node)
  sorted = topsort(node)
  for i = length(sorted):-1:1
    n = sorted[i]
    #backward!(n.value)
    println(typeof(n.f))
  end
  error("")
end

function topsort(node::Node)
  sorted = Node[]
  dict = ObjectIdDict()
  function visit(n::Node)
    c = get!(dict, n, 1)
    if c == 1
      for t in n.tails
        visit(t)
      end
      push!(sorted, n)
    end
  end
  visit(node)
  sorted
end
