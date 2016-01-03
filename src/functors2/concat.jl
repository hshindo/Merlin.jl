type Concat <: Functor
  dim::Int
  x::Vector{Variable}
  y::Variable
end

function forward!{T,N}(f::CrossEntropy{Array{T,N}})
  xs = map(xx -> xx.value, f.x)

end

function forward{T,N}(f::Concat, xs::Vector{Array{T,N}})
  sum = 0
  for x in xs
    sum += size(x, f.dim)
  end
  outsize = [size(xs[1])...]
  outsize[f.dim] = sum
  y = Array(T, outsize...)

  range = map(s -> 1:s, outsize)
  index = 1
  for x in xs
    s = size(x, f.dim)
    range[f.dim] = index:(index + s - 1)
    y[range...] = x
    index += s
  end
  y, (gy, gx) -> gx == nothing || backward!(f, gy, gx)
end

function backward!{T,N}(f::Concat, gy::Array{T,N}, gx::Vector{Array{T,N}})
  range = map(s -> 1:s, [size(gy)...])
  index = 1
  for i = 1:length(gx)
    s = size(gx[i], f.dim)
    range[f.dim] = index:(index + s - 1)
    axpy!(T(1), gy[range...], gx[i])
    index += s
  end
end
