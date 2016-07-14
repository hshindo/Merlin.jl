export SGD

"""
Stochastic Gradient Descent.
"""
type SGD
  rate::Float64
end

function update!{T}(opt::SGD, data::Array{T}, grad::Array{T})
  BLAS.axpy!(-T(opt.rate), grad, data)
  fill!(grad, T(0))
end
