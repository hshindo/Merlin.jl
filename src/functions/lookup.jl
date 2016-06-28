export lookup

"""
    Lookup(ws::Vector{Var})

Lookup function.

### 👉 Example
```julia
f = Lookup(Float32,10000,100) # 100-length vector, 10k vocabulary
# f = Lookup(Float32,10000,100, device=:CUDA)
x = Var(rand(1:1000,5,3))
y = f(x)
```
"""
type Lookup
  ws::Vector{Var}
end

"""
    Lookup{T}(::Type{T}, indim, outdim, [device])
"""
function Lookup{T}(::Type{T}, indim::Int, outdim::Int, device=:CPU)
  ws = Var[param(Vector{T}(randn(outdim))) for i=1:indim]
  if device == :CUDA
    for w in ws
      w.value = CuArray(w.value)
    end
  end
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
  @checkargs f (x,)
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

#=
function lookup2(w::Var, x::Var)
  y = lookup(w.value, x.value)
  df(gy) = ∇lookup!(w.value, w.grad, x.value, gy)

  args = Var[]
  for id in IntSet(x.value)
    push!(args, slice(w, :, id))
  end
  Var(y, df, [])
end

function lookup2(w, x::Array{Int})
  n = size(w, 1)
  y = similar(w, size(x,1)*n, size(x)[2:end]...)
  for i = 1:length(x)
    copy!(y, (i-1)*n+1, w, (x[i]-1)*n+1, n)
  end
  y
end

function ∇lookup2!{T}(w, gw, x::Array{Int}, gy::Array{T})
  n = size(w, 1)
  for i = 1:length(x)
    soffs = (i - 1) * n + 1
    doffs = (x[i] - 1) * n + 1
    BLAS.axpy!(n, T(1), pointer(gy,soffs), stride(gy,1), pointer(gw,doffs), stride(gw,1))
  end
end
=#
