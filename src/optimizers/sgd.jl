export SGD

"""
    SGD

Stochastic Gradient Descent.

## Arguments
* rate: learning rate
* momentum: momentum coefficient
"""
type SGD
    rate::Float64
    momentum::Float64
    states::ObjectIdDict
end

function SGD(rate; momentum=0.0)
    SGD(rate, momentum, ObjectIdDict())
end

function (opt::SGD){T}(data::AbstractArray{T}, grad::AbstractArray{T})
    if opt.momentum != 0.0
        state = get!(opt.states, data, nothing)
        if state == nothing
            v = zeros(data)
            opt.states[data] = v
        else
            v = state::Array{T}
        end
        BLAS.scal!(length(v), T(opt.momentum), v, 1)
        BLAS.axpy!(T(opt.rate), grad, v)
        broadcast!(-, data, data, v)
    else
        BLAS.axpy!(-T(opt.rate), grad, data)
    end
    fill!(grad, T(0))
end
