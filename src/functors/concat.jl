type Concat <: Functor
  dim::Int
end

function apply(fun::Concat, vars::Vector{Variable})
  input = map(v -> v.data, vars)
  sum = 0
  for x in input
    sum += size(x, fun.dim)
  end
  outsize = [size(input[1])...]
  outsize[fun.dim] = sum
  data = Array(eltype(input[1]), outsize...)

  range = map(s -> 1:s, outsize)
  index = 1
  for x in input
    s = size(x, fun.dim)
    range[fun.dim] = index:(index + s - 1)
    data[range...] = x
    index += s
  end
  Variable(data)
end

function diff(fun::Concat, tails::Vector{Variable}, head::Variable)
  head.data, head.work
end

#####
function apply(fun::Concat, input::Vector)
  sum = 0
  for x in input
    sum += size(x, fun.dim)
  end
  outsize = [size(input[1])...]
  outsize[fun.dim] = sum
  output = Array(eltype(input[1]), outsize...)

  range = map(s -> 1:s, outsize)
  index = 1
  for x in input
    s = size(x, fun.dim)
    range[fun.dim] = index:(index + s - 1)
    output[range...] = x
    index += s
  end
  (output,)
end

function diff(fun::Concat, input::Vector, gradout::Array)
  gradin = map(similar, input)
  range = map(s -> 1:s, [size(gradout)...])
  index = 1
  for g in gradin
    s = size(g, fun.dim)
    range[fun.dim] = index:(index + s - 1)
    g[:] = gradout[range...]
    index += s
  end
  gradin
end
