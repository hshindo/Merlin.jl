type Linear <: Functor
  weight::Variable
  bias::Variable
end

function Linear{T}(::Type{T}, inlen::Int, outlen::Int)
  weight = convert(Array{T}, randn(outlen, inlen) * sqrt(1 / inlen))
  bias = fill(T(0.01), outlen)
  w = Variable(weight, zeros(weight))
  b = Variable(bias, zeros(bias))
  Linear(w, b)
end

call(f::Linear, x::Variable) = f.weight * x + f.bias
