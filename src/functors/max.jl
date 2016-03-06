type Max <: Functor
  dim::Int
end

function forward!(f::Max, v::Variable)
  v.value, v.work = max(v[1].value, f.dim)
end

function max{T,N}(x::Array{T,N}, dim::Int)
  findmax(x, dim)
end

function backward!(f::Max, v::Variable)
  gx = ∇max(v[1].value, v.grad, v.work)
  addgrad!(v[1], gx)
end

function ∇max{T,N}(x::Array{T,N}, gy::Array{T,N}, idx::Array{Int,N})
  gx = zeros(x)
  for i = 1:length(idx)
    gx[idx[i]] = gy[i]
  end
  gx
end
