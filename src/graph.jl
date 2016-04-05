type Graph <: Functor
  vars::Vector{Variable} # sorted in bottom-up order
  inputids::Vector{Int}
end

function Graph(var::Variable)

end

function Graph(funs::Functor...)

end

function compile(var::Variable)
  # flatten add
  if typeof(var.f) == Add
    #any(a -> typeof(a.f) == Add, var.args)
    args = Variable[]
    for a in var.args
      typeof(a.f) == Add ? append!(args, a.args) : push!(args, a)
    end
    var.args = args
  end
end

function backward!(f::Graph, var::Variable)
  for i = length(f.vars):-1:1
    v = f.vars[i]

  end
end
