type Concat <: Functor
  dim::Int
end

function apply{T,N}(f::Concat, input::Vector{Array{T,N}})
  sum = 0
  for x in input
    sum += size(x, f.dim)
  end
  outsize = [size(input[1])...]
  outsize[f.dim] = sum
  output = Array(T, outsize...)

  range = map(s -> 1:s, outsize)
  index = 1
  for x in input
    s = size(x, f.dim)
    range[f.dim] = index:(index + s - 1)
    output[range...] = x
    index += s
  end
  output, gy -> diff(f, input, gy)
end

function diff{T,N}(fun::Concat, input::Vector{Array{T,N}}, gradout::Array{T,N})
  gradin = Array(Array{T,N}, length(input))
  range = map(s -> 1:s, [size(gradout)...])
  index = 1
  for i = 1:length(gradin)
    s = size(input[i], fun.dim)
    range[fun.dim] = index:(index + s - 1)
    gradin[i] = gradout[range...]
    index += s
  end
  gradin
end
