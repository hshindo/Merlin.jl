type Concat <: Functor
  dim::Int
end

function apply{T,N}(fun::Concat, inputs::Vector{Array{T,N}})
  sum = 0
  for x in inputs
    sum += size(x, fun.dim)
  end
  outsize = [size(inputs[1])...]
  outsize[fun.dim] = sum
  output = Array(T, outsize...)

  range = map(s -> 1:s, outsize)
  index = 1
  for x in inputs
    s = size(x, fun.dim)
    range[fun.dim] = index:(index + s - 1)
    output[range...] = x
    index += s
  end
  output, nothing
end

function diff{T,N}(fun::Concat, inputs::Vector{Array{T,N}}, gradout::Array{T,N})
  gradins = map(similar, inputs)
  range = map(s -> 1:s, [size(gradout)...])
  index = 1
  for g in gradins
    s = size(g, fun.dim)
    range[fun.dim] = index:(index + s - 1)
    g[:] = gradout[range...]
    index += s
  end
  gradins
end
