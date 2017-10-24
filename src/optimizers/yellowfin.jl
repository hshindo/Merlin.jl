export YellowFin

"""
    YellowFin

## References
Jian Zhang, Ioannis Mitliagkas, Christopher Ré, "YellowFin and the Art of Momentum Tuning", 2017
"""
mutable struct YellowFin
    alpha::Float64
    mu::Float64
    beta::Float64
end

YellowFin() = YellowFin(1.0, 0.0)

mutable struct YelloFinState
    hs::Vector
    hmax::Float64
    hmin::Float64
    g2::Array
    g::Array
end

wsum(beta, a, b) = beta * a + (1-beta) * b

function (opt::YellowFin){T}(x::Array{T}, gx::Array{T})
    curvature_range!(opt.state, opt.beta, gx)
    C = gradient_variance(opt.state, opt.beta, gx)
    D = distance_to_opt(opt.state, opt.beta, gx)

    sqrt_mu = solve(C, D, opt.state.hmin)
    alpha = (1-sqrt_mu) * (1-sqrt_mu) / hmin
    mu = sqrt_mu * sqrt_mu

    opt.mu = wsum(beta, opt.mu, mu)
    opt.alpha = wsum(beta, opt.alpha, alpha)
    momentum(grads, params, learning_rate_factor * updates[alpha], updates[mu], updates)
end

function solve(c::Float64, d::Float64, hmin::Float64)
    # Eq: x^2 D^2 + (1-x)^4 * C / h_min^2
    # where x = sqrt(mu)
    # Minimising this reduces to solving
    # y^3 + p * y + p = 0
    # y = x - 1
    # p = (D^2 h_min^2) / 2C
    p = d * d * hmin * hmin / 2C
    w3 = p * (sqrt(0.25 + p / 27.0) - 0.5)
    w = w3^(1/3)
    y = w - p/3w
    sqrt_mu1 = y + 1

    sqrt_mu2 = (sqrt(h_max)-sqrt(h_min)) / (sqrt(h_max)+sqrt(h_min))
    sqrt_mu = max(sqrt_mu1, sqrt_mu2)
end

function curvature_range!{T}(state::YelloFinState, beta::Float64, gx::Array{T})
    h = sumabs2(gx)
    hs = state.hs
    push!(hs, h)
    hmax = maximum(hs)
    hmin = minimum(hs)
    shift!(hs)
    state.hmax = beta * state.hmax - (1-beta) * hmax
    state.hmin = beta * state.hmin - (1-beta) * hmin
end

function gradient_variance{T}(state::YelloFinState, beta::Float64, gx::Array{T})
    state.g2 = beta * state.g2 + (1-beta) * (gx.*gx)
    state.g = beta * state.g + (1-beta) * gx
    abssum(state.g2 - abs2.(state.g))
end

function distance_to_opt(state, grad::Array, beta::Float64)

end

def distance_to_optim(grads, beta, updates):
    """
    Routine fro calculating the distance to the optimum.
    """
    # Had issue with initializing to 0.0, so switched to 1.0
    g = theano.shared(value_floatX(1.0), name="g")
    h = theano.shared(value_floatX(1.0), name="h")
    d = theano.shared(value_floatX(1.0), name="d")
    # L2 norm
    l2_norm = sum(T.sum(T.sqr(g)) for g in grads)
    updates[g] = ema(beta, g, T.sqrt(l2_norm))
    updates[h] = ema(beta, h, l2_norm)
    updates[d] = ema(beta, d, updates[g] / updates[h])
    return updates[d]
