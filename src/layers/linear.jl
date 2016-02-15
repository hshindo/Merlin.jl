type Linear <: Layer
  w::Variable
  b::Variable
  in::Layer
  out::Variable
end

function Linear{T}(::Type{T}, xlength::Int, ylength::Int)
  w = randn(ylength, xlength) * sqrt(1 / xlength)
  w = convert(Matrix{T}, w) |> AFArray
  b = fill(T(0.01), ylength) |> AFArray
  args = [Variable(w), Variable(b), Variable()]
  Linear(args, nothing)
end

function Linear(w::Variable, b::Variable, x::Layer)

end

function Base.call(f::Linear, x::Variable)
  w, b = f.w, f.b
  y = w.value * x + b.value
  Linear([w,b,x], y)
end

function backward(f::Linear)

end
