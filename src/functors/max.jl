type Max <: Functor
  dim::Int
end

function forward!(f::Max, v::Variable)
  y, idx = max(f.dim, v[1].value)
  v.value = y
  v.backward! = () -> ∇max!(idx, v[1].grad, v.grad)
end

function max{T,N}(dim::Int, x::Array{T,N})
  findmax(x, dim)
end

function ∇max!{T,N}(idx::Array{Int,N}, gx::Array{T,N}, gy::Array{T,N})
  for i = 1:length(idx)
    gx[idx[i]] += gy[i]
  end
end
