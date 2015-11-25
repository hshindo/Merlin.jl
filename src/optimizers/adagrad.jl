"""
AdaGrad implementation.
See: http://jmlr.org/papers/v12/duchi11a.html
"""
type AdaGrad <: Optimizer
  alpha::Float64
  states::ObjectIdDict
end

AdaGrad(alpha::Float64) = AdaGrad(alpha, ObjectIdDict())

function update!{T}(opt::AdaGrad, param::Array{T}, grad::Array{T})
  state = get!(opt.states, param, nothing)
  if state == nothing
    sqgrad = zeros(T, length(param))
    opt.states[param] = sqgrad
  else
    sqgrad = state::Array{T}
  end
  for i = 1:length(grad)
    sqgrad[i] += grad[i] * grad[i]
    param[i] -= T(opt.alpha) * grad[i] / (sqrt(sqgrad[i]) + T(1e-8))
  end
  fill!(grad, T(0.0))
end
