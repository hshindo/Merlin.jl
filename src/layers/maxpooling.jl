type MaxPooling <: Layer
  dim::Int
  idx
  args
  out

  function MaxPooling(dim::Int)
    l = MaxPooling(dim, nothing, nothing, nothing)
    finalizer(l, free)
    l
  end
end

function free(l::MaxPooling)
  release(l.out.value)
end

function forward!(l::MaxPooling)
  x = l.args[1].value
  y, idx = findmax(x, dim)
  l.out.value = y
  l.idx = idx
end
