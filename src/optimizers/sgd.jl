type SGD <: Optimizer
  learnrate::Float64
end

function update!{T}(opt::SGD, param::Array{T}, grad::Array{T})
  axpy!(-T(opt.learnrate), grad, param)
  fill!(grad, T(0.0))
end
