type Pooling <: Functor
end

clone(fun::Pooling) = fun

function apply{T}(fun::Pooling, input::Matrix{T})
  output = Array(T, size(input, 1), 1)
  if size(input, 2) == 1
    output[:] = input[:]
  else
    m = maximum(input, 2) # inefficient
    output[:] = m[:]
  end
  output
end

function diff{T}(fun::Pooling, gradout::Matrix{T}, input::Matrix{T})
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
  tuple(gradin)
end
