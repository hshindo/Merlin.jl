export AdaGrad

"""
    AdaGrad

AdaGrad Optimizer.

# References
* Duchi t al., ["Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf), JMLR 2011.
"""
mutable struct AdaGrad
    alpha::Float64
    states::IdDict
end

AdaGrad(alpha::Float64) = AdaGrad(alpha, IdDict())

function (opt::AdaGrad)(value::Array{T}, grad::Array{T}) where T
    state = get!(opt.states, value, nothing)
    if state == nothing
        sqgrad = zeros(T, length(value))
        opt.states[value] = sqgrad
    else
        sqgrad = state::Array{T}
    end
    @inbounds for i = 1:length(grad)
        sqgrad[i] += grad[i] * grad[i]
        if abs(sqgrad[i]) > T(1e-8)
            value[i] -= T(opt.alpha) * grad[i] / sqrt(sqgrad[i])
        end
    end
    fill!(grad, T(0.0))
end
(opt::AdaGrad)(x::Var) = opt(x.data, x.grad)
