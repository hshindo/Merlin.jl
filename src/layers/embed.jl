export Embed

type Embed <: Var
  data
  grad
  tails::Vector{Var}
  idset::IntSet
end

function Embed(w::Var, x::Var)
  data = lookup(w.data, x.data)
  Embed(data, nothing, [w,x], IntSet())
end

function Embed{T}(::Type{T}, indim::Int, outdim::Int)
  Embed(randn(T,outdim,indim), Data())
end

@compat (l::Embed)(x::Var) = Embed(l.w, x)

function lookup{T}(w, x::Array{Int})
  n = size(w, 1)
  s = Int[size(x,i) for i=1:ndims(x)]
  s[1] *= n
  y = similar(w, s...)
  for i = 1:length(x)
    copy!(y, (i-1)*n+1, w, (x[i]-1)*n+1, n)
  end
  y
end

function update!(v::Embed, opt)
  for id in v.idset
    update!(opt)
  end
  empty!(v.idset)
end
