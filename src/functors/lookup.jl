export lookup

"""
Lookup function.

### 👉 Example
```julia
ws = [Var(rand(Float32,100)) for i=1:10000] # 100-length vector, 10k vocabulary
x = Var(rand(1:1000,5,3))
y = lookup(w, x)
```
"""
function lookup(ws::Vector{Var}, x::Var)
  y = lookup(w.value, x.value)
  f(gy) = ∇lookup!(w.value, w.grad, x.value, gy)
  Var(y, nothing, f, ws)
end

function lookup{T}(w::Matrix{T}, x::Array{Int})
  s = Int[size(x)...]
  s[1] *= size(w, 1)
  y = Array(T, s...)
  n = size(w, 1)
  for i = 1:length(x)
    copy!(y, (i-1)*n+1, w, (x[i]-1)*n+1, n)
  end
  y
end

function lookup{T}(w::CuMatrix{T}, x::CuMatrix{Int})

end

function ∇lookup!{T}(w::Matrix{T}, gw::Matrix{T}, x::Array{Int}, gy::Matrix{T})
  n = size(w, 1)
  for i = 1:length(x)
    BLAS.axpy!(T(1), gy, slice(w,:,i))
  end
end

function ∇lookup!{T}(w::CuMatrix{T}, gw::CuMatrix{T}, x::CuArray{Int}, gy::CuMatrix{T})

end
