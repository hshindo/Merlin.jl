export AdaGrad, Adam, SGD

"""
    AdaGrad

AdaGrad Optimizer.

# References
* Duchi t al., ["Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf), JMLR 2011.
"""
mutable struct AdaGrad
    alpha::Float64
    states::ObjectIdDict
end

AdaGrad(alpha::Float64) = AdaGrad(alpha, ObjectIdDict())

(opt::AdaGrad)(x::Var) = opt(x.data, x.grad)
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

"""
    Adam

Adam Optimizer

# References
* Kingma and Ba, ["Adam: A Method for Stochastic Optimization"](http://arxiv.org/abs/1412.6980v8), ICLR 2015.
"""
mutable struct Adam
    alpha::Float64
    beta1::Float64
    beta2::Float64
    eps::Float64
    states::ObjectIdDict
end

Adam() = Adam(0.001, 0.9, 0.999, 1e-8, ObjectIdDict())
Adam(alpha)= Adam(alpha, 0.9, 0.999, 1e-8, ObjectIdDict())

(opt::Adam)(x::Var) = opt(x.data, x.grad)
function (opt::Adam)(param::Array{T}, grad::Array{T}) where T
    @assert length(param) == length(grad)
    state = get!(opt.states, param, nothing)
    if state == nothing
        m, v, t = zeros(param), zeros(grad), 1
    else
        m::Array{T}, v::Array{T}, t::Int = state
    end
    BLAS.scale!(T(opt.beta1), m)
    BLAS.axpy!(T(1.0 - opt.beta1), grad, m)
    BLAS.scale!(T(opt.beta2), v)
    @inbounds for i = 1:length(grad)
        grad[i] = grad[i] * grad[i]
    end
    BLAS.axpy!(T(1.0 - opt.beta2), grad, v)
    fix1 = 1.0 - opt.beta1 ^ t
    fix2 = 1.0 - opt.beta2 ^ t
    rate = opt.alpha * sqrt(fix2) / fix1
    @inbounds for i = 1:length(param)
        param[i] -= rate * m[i] / (sqrt(v[i]) + T(opt.eps))
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
(opt::Adam)(x::Var) = opt(x.data, x.grad)

"""
    SGD

Stochastic Gradient Descent Optimizer.

# Arguments
* rate: learning rate
* [momentum=0.0]: momentum coefficient
* [nesterov=false]: use nesterov acceleration or not
"""
mutable struct SGD
    rate::Float64
    momentum::Float64
    nesterov::Bool
    states::ObjectIdDict
end

function SGD(rate=0.0; momentum=0.0, nesterov=false)
    SGD(rate, momentum, nesterov, ObjectIdDict())
end

function (opt::SGD)(x::Array{T,N}, gx::Array{T,N}) where {T,N}
    if opt.momentum > 0.0
        if haskey(opt.states, x)
            v = opt.states[x]
        else
            v = zeros(x)
            opt.states[x] = v
        end
        m = T(opt.momentum)
        rate = T(opt.rate)
        v .= m .* v - rate * gx
        if opt.nesterov
            v = copy(v)
            BLAS.scal!(length(v), m, v, 1)
            BLAS.axpy!(-rate, gx, v)
        end
        BLAS.axpy!(T(1), v, x)
    else
        # BLAS.axpy!(T(0.0005), x, gx)
        BLAS.axpy!(T(-opt.rate), gx, x)
    end
    fill!(gx, T(0))
end
(opt::SGD)(x::Var) = opt(x.data, x.grad)
