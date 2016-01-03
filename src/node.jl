type Node
  value::Functor
  tails::Vector{Node}
end

Node(value) = Node(value, Node[])

function call(f::Functor, arg::Node)
  forward!(f, arg.value.y)
  Node(f, [arg])
end

function compile(node::Node{Functor})
  topsort(node)
end

function backward!(node::Node{Functor})
  sorted = topsort(node)
  for i = length(sorted):-1:1
    n = sorted[i]
    backward!(n.value)
  end
end

function topsort(node::Node)
  sorted = Var[]
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

type Graph <: Functor
  nodes::Vector{Node}
end

function forward!(f::Graph, x)
  for n in f.nodes

  end
end

function forward!()
end
