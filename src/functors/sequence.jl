type Sequence <: Functor
  funs::Vector{Functor}
end

Sequence(funs::Functor...) = Sequence([funs...])

function apply(seq::Sequence, var::Variable)
  for fun in seq.funs
    var = var |> fun
  end
  var
end

function diff!(seq::Sequence, var::Variable)
  diff!(seq.funs[end], var)
  #for i = length(seq.funs):-1:1
  #  diff!(seq.funs[i], var)
  #  var = var.tails[1]
  #end
end

function apply2(seq::Sequence, input)
  outputs = []
  for fun in seq.funs
    output = apply(fun, input)
    push!(outputs, output)
    input = output[1]
  end
  input, outputs
end

function diff2(seq::Sequence, input, work::Vector, gradout)
  for i = length(seq.funs):-1:1
    fun = seq.funs[i]
    diff(fun, input, gradout)
  end
end
