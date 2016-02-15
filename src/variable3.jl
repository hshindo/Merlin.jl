type Variable
  args::Vector{Layer}
  value
  grad
end

Variable(args, value) = Variable(args, value, nothing)
Variable(value) = Variable(Layer[], value)
Variable() = Variable(Layer[], nothing, nothing)
