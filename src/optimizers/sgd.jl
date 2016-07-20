export SGD

"""
    SGD

Stochastic Gradient Descent.

* rate: learning rate
"""
type SGD
  rate::Float64
end

@compat function (opt::SGD){T}(data::Array{T}, grad::Array{T})
  BLAS.axpy!(-T(opt.rate), grad, data)
  fill!(grad, T(0))
end
