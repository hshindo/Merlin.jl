type Graph <: Functor
  vars::Vector{Variable} # sorted in bottom-up order
end

function Graph(var::Variable)

end

function forward!(f::Graph)
  for v in f.vars

  end
end
