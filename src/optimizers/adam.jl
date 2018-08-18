export Adam

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
    states::IdDict
end

Adam() = Adam(0.001, 0.9, 0.999, 1e-8, IdDict())
Adam(alpha)= Adam(alpha, 0.9, 0.999, 1e-8, IdDict())

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
(opt::Adam)(x::Var) = opt(x.data, x.grad)

function update_slow!(opt::Adam, param::Array{T}, grad::Array{T}) where T
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
