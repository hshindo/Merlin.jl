export lookup

type Lookup; end

function Lookup{T}(::Type{T}, insize::Int, outsize::Int)
  ws = Array(Var, insize)
  for i = 1:insize
    w = convert(Vector{T}, randn(outsize))
    ws[i] = param(w)
  end
  Lookup(ws)
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

"""
Lookup function.

### 👉 Example
```julia
ws = [Var(rand(Float32,100)) for i=1:10000] # 100-length vector, 10k vocabulary
x = rand(1:1000,5,3)
y = lookup(w, x)
```
"""
function lookup(ws::Vector{Var}, x::Array{Int})
  xs = map(id -> ws[id], x)
  args = Var[]
  for id in IntSet(x)
    push!(args, ws[id])
  end
  y = concat(1, xs)
  dims = Int[size(x)...]
  dims[1] *= n = size(length(ws[1].value), 1)
  y = reshape(y, dims...)
  f(gy) = ∇lookup!(ws, x, gy)
  Var(y, f, args)
end

#=
function lookup(ws::Vector{Var}, x::Array{Int})
  s = Int[size(x)...]
  s[1] *= size(w, 1)
  y = Array(T, s...)
  n = size(w, 1)
  for i = 1:length(x)
    copy!(y, (i-1)*n+1, ws[x[i]].value, (x[i]-1)*n+1, n)
  end
  y
end
=#

function ∇lookup!{T}(ws::Vector{Var}, x::Array{Int}, gy::Array{T})
  len = length(ws[1].value)
  offset = 1
  for i = 1:length(x)
    gw = ws[x[i]].grad
    BLAS.axpy!(len, T(1), pointer(gy,offset), stride(gy,1), pointer(gw), stride(gw,1))
    offset += len
  end
end

function ∇lookup!{T}(w::Matrix{T}, gw::Matrix{T}, x::Array{Int}, gy::Matrix{T})
  n = size(w, 1)
  for i = 1:length(x)
    BLAS.axpy!(T(1), gy, slice(w,:,i))
  end
end
