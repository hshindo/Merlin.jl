export Embed

type Embed <: Var
  data
  grad
  tails::Vector{Var}
  idset::IntSet
end

function Embed(w::Var, x::Var)
  (hasdata(w) && hasdata(x)) || return Embed(nothing, nothing, [w,x], IntSet())
  data = lookup(w.data, x.data)
  Embed(data, nothing, [w,x], IntSet())
end

"""
    Embed{T}(::Type{T}, indim, outdim, [device])

### 👉 Example
```julia
f = Embed(Float32,10000,100) # 100-length vector, 10k vocabulary
x = Var(rand(1:1000,5,3))
y = f(x)
```
"""
function Embed{T}(::Type{T}, indim::Int, outdim::Int)
  Embed(Param(randn(T,outdim,indim)), Data())
end

function Embed{T}(path, ::Type{T})
  lines = open(readlines, path)
  ws = Array(Var, length(lines))
  for i = 1:length(lines)
    items = split(chomp(lines[i]), ' ')
    w = map(x -> parse(T,x), items)
    ws[i] = param(w)
  end
  Embed(ws)
end

@compat (v::Embed)(x::Var) = Embed(v[1], x)
@compat (v::Embed)(w::Var, x::Var) = Embed(w, x)

function lookup(w, x::Array{Int})
  n = size(w, 1)
  s = Int[size(x,i) for i=1:ndims(x)]
  s[1] *= n
  y = similar(w, s...)
  for i = 1:length(x)
    copy!(y, (i-1)*n+1, w, (x[i]-1)*n+1, n)
  end
  y
end

backward!(v::Embed) = ∇lookup!()

function ∇lookup!(w, x::Array{Int}, gy)
  T = eltype(gy)
  n = size(w, 1)
  for i = 1:length(x)
    BLAS.axpy!(n, T(1), pointer(gy,(i-1)*n+1), 1, pointer(gx,(x[1]-1)*n+1), 1)
  end
end

function update!(opt, v::Embed)
  for id in v.idset
    update!(opt, v.data, v.grad)
  end
  empty!(v.idset)
end
