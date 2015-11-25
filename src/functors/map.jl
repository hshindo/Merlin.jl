type Map <: Functor
  fun::Functor
end

function apply(fun::Map, var::Variable)
  var.data <: Vector && error("Map: data is not vector")
  output = map(x -> apply(fun.fun, x), var.data)
  Variable(output)
end

function diff()

end
