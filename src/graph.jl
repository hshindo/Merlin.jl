type Graph <: Functor
  vars::Vector{Variable} # sorted in bottom-up order
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

@compat function (g::Graph)(args::Tuple)
  x = arg
  @assert (length(g.vars) == length(args))
  for i = 1:length(g.vars)
    g.vars[i]
  end
end

function update!(opt::Optimizer, seq::Sequence)
  for f in seq.funs
    applicable(update!, opt, f) && update!(opt, f)
  end
end
