type Concat <: Functor
  dim::Int
end

clone(fun::Concat) = fun

function apply{T,N}(fun::Concat, inputs::Array{T,N}...)
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
  output
end

function diff{T,N}(fun::Concat, gradout::Array{T,N}, inputs::Array{T,N}...)
  range = map(s -> 1:s, [size(gradout)...])
  index = 1
  i = 1
  map(inputs) do x
    s = size(x, fun.dim)
    range[fun.dim] = index:(index + s - 1)
    index += s
    gradout[range...]
  end
end
