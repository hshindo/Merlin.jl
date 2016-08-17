export AdaGrad

"""
AdaGrad implementation.
See: http://jmlr.org/papers/v12/duchi11a.html
"""
type AdaGrad
  alpha::Float64
  states::ObjectIdDict
end

AdaGrad(alpha::Float64) = AdaGrad(alpha, ObjectIdDict())

function (opt::AdaGrad){T}(value::Array{T}, grad::Array{T})
  state = get!(opt.states, value, nothing)
  if state == nothing
    sqgrad = zeros(T, length(value))
    opt.states[value] = sqgrad
  else
    sqgrad = state::Array{T}
  end
  @inbounds @simd for i = 1:length(grad)
    sqgrad[i] += grad[i] * grad[i]
    value[i] -= T(opt.alpha) * grad[i] / (sqrt(sqgrad[i]) + T(1e-8))
  end
  fill!(grad, T(0.0))
end
