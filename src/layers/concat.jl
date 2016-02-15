type Identity <: Layer
  out::Variable
end

Identity(in::Variable) = Identity(in)

type Concat <: Layer
  ins::Vector{Variable}
  dim::Int
  out::Variable
end

function Concat(ins::Vector, dim::Int)
  xs = map(in -> in.out.value, ins)
  y = cat(xs, dim)
  Concat(ins, dim, Variable(y))
end

function backward(l::Concat)
  gy = in.value
  offset = 0
  for i = 1:len
    x = v[i].value
    s = size(x, f.dim)
    indices = AFArray([offset:offset+s-1])
    gx = lookup(gy, indices, f.dim)
    addgrad!(v[i], gx)
    offset += s
  end
end

type Lookup <: Layer
  weight::Variable
  idset::Set{Int}
  out::Variable
end

function Lookup{T}(::Type{T}, size1::Int, size2::Int)
  w = randn(size1, size2)
  w = convert(Matrix{T}, w) |> AFArray
  Lookup(Variable(w), Set{Int}(), Variable())
end

function Base.call(f::Lookup, in::Layer)
  x = in.out
  y = lookup(f.weight.value, x.value, 2)
  Lookup(f.weight, f.idset, Variable([in],y))
end

type MaxPool2D <: Layer
  dim::Int
  idx
  out::Variable
end

function MaxPool2D(in::Layer, dim::Int)
  x = in.out.value
  y, idx = findmax(x, dim)
  MaxPool2D(dim, idx, Variable([in],y))
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
