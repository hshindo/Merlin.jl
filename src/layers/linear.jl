type Linear <: Layer
  w::Variable
  b::Variable
  out::Variable

  function Linear(w, b)
    l = Linear(w, b, nothing, nothing)
    finalizer(l, free)
    l
  end
end

function Linear{T}(::Type{T}, size1::Int, size2::Int)
  w = randn(size2, size1) * sqrt(1 / size1)
  w = convert(Matrix{T}, w) |> AFArray
  b = fill(AFArray, T(0.01), size2)
  Linear(Variable(w), Variable(b))
end

function forward!(l::Linear)
  x = v[1].value
  w, b = f.w.value, f.b.value
  v.value = w * x + b
end

function free(l::Linear)
  release(l.w)
  release(l.b)
  release(l.out)
end
