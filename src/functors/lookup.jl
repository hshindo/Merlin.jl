export lookup

type Lookup
  ws::Vector{Var}
end

function Lookup{T}(::Type{T}, insize::Int, outsize::Int)
  ws = Array(Var, insize)
  for i = 1:insize
    w = convert(Vector{T}, randn(outsize))
    ws[i] = Var(w, grad=zeros(w))
  end
  Lookup(ws)
end

function Lookup{T}(path, ::Type{T})
  lines = open(readlines, path)
  ws = Array(Var, length(lines))
  for i = 1:length(lines)
    items = split(chomp(lines[i]), ' ')
    w = map(x -> parse(T,x), items)
    ws[i] = Var(w, zeros(w))
  end
  Lookup(ws)
end

"""
Lookup variables and concat.

### 👉 Example
```julia
f = Lookup(Float32, 1000, 100)
x = Var(rand(1:1000,5,3))
y = f(x)
```
"""
@compat function (f::Lookup)(x::Var)
  y = lookup(f.ws, x.value)
  args = map(id -> w[id], x.val)
  Var(y, f, args, ∇lookup!)
end

function lookup(ws::Vector{Var}, x::Matrix{Int})
  w1 = ws[1].value
  len = length(w1)
  T = eltype(w1)
  y = Array(T, len*size(x,1), size(x,2))
  offset = 1
  for i = 1:length(x)
    copy!(y, offset, ws[x[i]].value, 1, len)
    offset += len
  end
  y
end

function ∇lookup!(y::Var)
  x = y[1]
  ∇lookup!(y.f.ws, x.value, y.grad)
end

function ∇lookup!{T}(ws::Vector{Var}, x::Matrix{Int}, gy::Matrix{T})
  len = length(ws[1].val)
  offset = 1
  for i = 1:length(x)
    gw = ws[x[i]].grad
    BLAS.axpy!(len, T(1), pointer(gy,offset), stride(gy,1), pointer(gw), stride(gw,1))
    offset += len
  end
end
