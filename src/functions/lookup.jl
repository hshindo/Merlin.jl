export Lookup

"""
    Lookup(ws::Vector{Var})

Lookup function.

### 👉 Example
```julia
f = Lookup(Float32,10000,100) # 100-length vector, 10k vocabulary
# f = Lookup(CuArray{Float32},10000,100)
x = Var(rand(1:1000,5,3))
y = f(x)
```
"""
type Lookup
  ws::Vector{Var}
end

function Lookup{T}(::Type{T}, indim::Int, outdim::Int, device=:CPU)
  ws = Var[param(Vector{T}(randn(outdim))) for i=1:indim]
  Lookup(ws)
end

"""
    Lookup{T}(path, ::Type{T})
"""
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

@compat function (f::Lookup)(x::Var)
  ws = f.ws
  ids = x.value
  args = Var[]
  vars = map(id -> ws[id], ids)
  for id in IntSet(ids)
    push!(args, ws[id])
  end
  y = lookup(ws, ids)
  df(gy) = ∇lookup!(ws, ids, gy)
  Var(y, df, args)
end

function lookup(ws::Vector{Var}, x::Array{Int})
  T = eltype(ws[1].value)
  n = length(ws[1].value)
  y = similar(ws[1].value, size(x,1)*n, size(x)[2:end]...)
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
