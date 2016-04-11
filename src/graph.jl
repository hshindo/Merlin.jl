type Graph <: Functor
  vars::Vector{Variable} # sorted in bottom-up order
end

function Graph(var::Variable)

end

function Graph(funs::Functor...)

end

function compile(var::Variable)

end

@compat function (g::Graph)(args::Tuple)
  x = arg
  @assert (length(g.vars) == length(args))
  for i = 1:length(g.vars)
    g.vars[i]
  end
end
