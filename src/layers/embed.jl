type Embed <: Layer
  w
  x
  y
  gy
  idset::IntSet
end

function Embed{T}(::Type{T}, indim::Int, outdim::Int)
  w = randn(T, outdim, indim)
  Embed(w, nothing, nothing, nothing, IntSet())
end

tails(l::Embed) = []

@compat function (l::Embed)(x::Layer)
  y = lookup(l, x.y)
  Embed(l.w, x.y, y, nothing, IntSet())
end
@compat (l::Embed)(x::GraphNode) = GraphNode(l, l.w, x)

function lookup(l::Embed, x::Array{Int})
  w = l.w.y
  n = size(w, 1)
  s = Int[size(x)...]
  s[1] *= n
  y = similar(w, s...)
  for i = 1:length(x)
    copy!(y, (i-1)*n+1, w, (x[i]-1)*n+1, n)
  end
  y
end

function update!(l::Embed, opt)
  for id in l.idset
    update!(opt)
  end
  empty!(l.idset)
end
