type Graph <: Functor
  vars::Vector{Variable} # sorted in bottom-up order
end

function Graph(var::Variable)

end

function Graph(funs::Functor...)

end

function call(f::Graph, out::Variable)
  for v in f.vars

  end
end

function backward!(f::Graph, var::Variable)
  for i = length(f.vars):-1:1
    v = f.vars[i]

  end
end
