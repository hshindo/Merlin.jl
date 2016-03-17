type Concat <: Functor
  dim::Int
end

function forward!(f::Concat, v::Variable)
  v.value = concat(f.dim, map(a -> a.value, v.args))
  v.backward! = () -> ∇concat!(f.dim, map(a -> a.grad, v.args), v.grad)
end

function concat{T,N}(dim::Int, xs::Vector{Array{T,N}})
  sum = 0
  for x in xs
    sum += size(x, dim)
  end
  outsize = [size(xs[1])...]
  outsize[dim] = sum
  y = Array(T, outsize...)

  range = map(s -> 1:s, outsize)
  index = 1
  for x in xs
    s = size(x, dim)
    range[dim] = index:(index + s - 1)
    y[range...] = x
    index += s
  end
  y
end

function concat{T,N}(dim::Int, xs::Vector{CudaArray{T,N}})

end

function ∇concat!{T,N}(dim::Int, gxs::Vector{Array{T,N}}, gy::Array{T,N})
  range = map(s -> 1:s, [size(gy)...])
  index = 1
  for i = 1:length(gxs)
    gx = gxs[i]
    s = size(gx, dim)
    range[dim] = index:(index + s - 1)
    axpy!(1.0, gy[range...], gx)
    index += s
  end
end

function ∇concat!{T,N}(dim::Int, xs::Vector{CudaArray{T,N}}, gy::Array{T,N})

end
