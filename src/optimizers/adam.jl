export Adam

"""
    Adam

Adam: A Method for Stochastic Optimization
See: http://arxiv.org/abs/1412.6980v8
"""
type Adam
  alpha::Float64
  beta1::Float64
  beta2::Float64
  eps::Float64
  states::ObjectIdDict
end

Adam() = Adam(0.001, 0.9, 0.999, 1e-8, ObjectIdDict())

function (opt::Adam){T}(param::Array{T}, grad::Array{T})
  @assert length(param) == length(grad)
  state = get!(opt.states, param, nothing)
  if state == nothing
    m, v, t = zeros(param), zeros(grad), 1
  else
    m::Array{T}, v::Array{T}, t::Int = state
  end
  scale!(T(opt.beta1), m)
  axpy!(T(1.0 - opt.beta1), grad, m)
  scale!(T(opt.beta2), v)
  for i = 1:length(grad)
    @inbounds grad[i] = grad[i] * grad[i]
  end
  axpy!(T(1.0 - opt.beta2), grad, v)
  fix1 = 1.0 - opt.beta1 ^ t
  fix2 = 1.0 - opt.beta2 ^ t
  rate = opt.alpha * sqrt(fix2) / fix1
  for i = 1:length(param)
    @inbounds param[i] -= rate * m[i] / (sqrt(v[i]) + T(opt.eps))
  end
  opt.states[param] = (m, v, t + 1)
  fill!(grad, T(0.0))
end

function update_slow!{T}(opt::Adam, param::Array{T}, grad::Array{T})
  state = get!(opt.states, param, nothing)
  if state == nothing
    m, v, t = zeros(param), zeros(grad), 1
  else
    m::Array{T}, v::Array{T}, t::Int = state
  end
  m += (1.0 - opt.beta1) * (grad - m)
  v += (1.0 - opt.beta2) * (grad .* grad - v)
  fix1 = 1.0 - opt.beta1 ^ t
  fix2 = 1.0 - opt.beta2 ^ t
  rate = opt.alpha * sqrt(fix2) / fix1
  for i = 1:length(param)
    param[i] -= rate * m[i] / (sqrt(v[i]) + T(opt.eps))
  end
  opt.states[param] = (m, v, t + 1)
  fill!(grad, T(0.0))
end
