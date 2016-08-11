export SGD

"""
    SGD

Stochastic Gradient Descent.

* rate: learning rate
"""
type SGD
    rate::Float64
    momentum::Float64
    states::ObjectIdDict
end

function SGD(rate; momentum=0.0)
    SGD(rate, momentum, ObjectIdDict())
end

@compat function (opt::SGD){T}(data::Array{T}, grad::Array{T})
    BLAS.axpy!(-T(opt.rate), grad, data)
    fill!(grad, T(0))
end
