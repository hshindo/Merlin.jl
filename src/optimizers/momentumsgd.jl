export MomentumSGD

"""
    Momentum SGD

v = mu * v + rate * grad
theta = theta - v
"""
type MomentumSGD
    rate::Float64
    mu::Float64 # momentum coefficient
    states::ObjectIdDict
end

MomentumSGD(rate=0.01, mu=0.9) = MomentumSGD(rate, mu, ObjectIdDict())

@compat function (opt::MomentumSGD){T}(value::Array{T}, grad::Array{T})
    state = get!(opt.states, value, nothing)
    if state == nothing
        v = zeros(value)
        opt.states[value] = v
    else
        v = state
    end

    BLAS.scal!(length(v), T(opt.mu), v, 1)
    BLAS.axpy!(T(opt.rate), grad, v)
    broadcast!(-, value, value, v)
    fill!(grad, T(0))
end
