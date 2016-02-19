type Variable
  value
  grad
end

Variable() = Variable(nothing)
Variable(value) = Variable(value, nothing, [])

function Base.call(l::Layer. deps::Vector{Layer})
  l.deps = deps
  forward!(l)
  l
end
Base.call(l::Layer, dep::Layer) = call(l, [dep])

type Sequence <: Layer
  ls::Vector{Layer}
end

Sequence(ls::Layer...) = Sequence([ls...])

