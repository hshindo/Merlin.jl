export Lookup

"""
Lookup function.

### 👉 Example
```julia
f = Lookup(Float32,10000,100) # 100-length vector, 10k vocabulary
x = rand(1:1000,5,3)
y = f(x)
```
"""
type Lookup
  ws::Vector{Var}
end

function Lookup{T}(path, ::Type{T})
  lines = open(readlines, path)
  ws = Array(Var, length(lines))
  for i = 1:length(lines)
    items = split(chomp(lines[i]), ' ')
    w = map(x -> parse(T,x), items)
    ws[i] = param(w)
  end
  Lookup(ws)
end

@compat function (f::Lookup)(args::Vector{Var})
  x = args[1]
  ws = f.ws
  xs = map(id -> ws[id], x.value)
  args = Var[]
  for id in IntSet(x.value)
    push!(args, ws[id])
  end
  y = lookup(ws, x.value)
  df(gy) = ∇lookup!(ws, x.value, gy)
  Var(y, df, args)
end
@compat (f::Lookup)(x::Var) = forward(f, [x])

function lookup(ws::Vector{Var}, x::Array{Int})
  T = eltype(ws[1].value)
  n = length(ws[1].value)
  s = Int[size(x)...]
  s[1] *= n
  y = Array(T, s...)
  for i = 1:length(x)
    copy!(y, (i-1)*n+1, ws[x[i]].value, 1, n)
  end
  y
end

function ∇lookup!{T}(ws::Vector{Var}, x::Array{Int}, gy::Array{T})
  n = length(ws[1].value)
  offset = 1
  for i = 1:length(x)
    gw = ws[x[i]].grad
    BLAS.axpy!(n, T(1), pointer(gy,offset), stride(gy,1), pointer(gw), stride(gw,1))
    offset += n
  end
end

function ∇lookup!{T}(w::Matrix{T}, gw::Matrix{T}, x::Array{Int}, gy::Matrix{T})
  n = size(w, 1)
  for i = 1:length(x)
    BLAS.axpy!(T(1), gy, slice(w,:,i))
  end
end
