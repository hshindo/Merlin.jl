type Pooling <: Functor
end

function apply(fun::Pooling, var::Variable)
  input = var.data
  output = Array(eltype(input), size(input, 1), 1)
  if size(input, 2) == 1
    output[:] = input[:]
  else
    m = maximum(input, 2) # inefficient
    output[:] = m[:]
  end
  Variable(output)
end

function apply(fun::Pooling, input::Matrix)
  output = Array(eltype(input), size(input, 1), 1)
  if size(input, 2) == 1
    output[:] = input[:]
  else
    m = maximum(input, 2) # inefficient
    output[:] = m[:]
  end
  (output,)
end

function diff(fun::Pooling, input::Matrix, gradout::Matrix)
  gradin = similar(input)
  if size(input, 2) == 1
    gradin[:] = gradout[:]
  else
    fill!(gradin, 0.0)
    _, inds = findmax(input, 2)
    s = size(input)
    map!(i -> ind2sub(s, i)[2], inds)
    for i = 1:length(inds)
      gradin[i, inds[i]] = gradout[i]
    end
  end
  gradin
end
