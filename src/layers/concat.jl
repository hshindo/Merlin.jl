type Concat <: Layer
  dim::Int
  var

  function Concat(dim)
    l = Concat(dim, nothing, nothing)
    finalizer(l, free)
    l
  end
end

function free(l::Concat)
  release(l.var)
end

function forward!(l::Concat)
  var = l.var
  xs = map(d -> d.var.value, var.deps)
  var.value = cat(xs, dim)
end

type ReLU <: Layer
  out::Variable
end

function ReLU(in::Layer)
  x = in.out.value
  wh = x >= 0.0
  y = wh .* x
  ReLU(Variable([in],y))
end

type Reshape <: Layer
  dims
  out::Variable
end

function Reshape(in::Layer, dims::Int...)
  x = in.out.value
  y = reshape(x, dims)
  Reshape(dims, Variable([in],y))
end

type Window2D <: Layer
  out::Variable
end

function Window2D(in::Layer, w1, w2, s1, s2, p1=0, p2=0)
  x = in.out.value
  y = unwrap(x, w1, w2, s1, s2, p1, p2)
  Window2D(Variable([in],y))
end
