export YellowFin

"""
    YellowFin

Jian Zhang, Ioannis Mitliagkas, Christopher Ré, "YellowFin and the Art of Momentum Tuning", 2017

"""
mutable struct YellowFin
    alpha::Float64
    mu::Float64
end

YellowFin() = YellowFin(1.0, 0.0)

function (opt::YellowFin)(x::Array{T,N}, gx::Array{T,N}) where {T,N}
    if opt.momentum > 0.0
        if haskey(opt.states, x)
            v = opt.states[x]
        else
            v = zeros(x)
            opt.states[x] = v
        end
        m = T(opt.momentum)
        rate = T(opt.rate)
        BLAS.scal!(length(v), m, v, 1)
        BLAS.axpy!(-rate, gx, v)
        if opt.nesterov
            v = copy(v)
            BLAS.scal!(length(v), m, v, 1)
            BLAS.axpy!(-rate, gx, v)
        end
        BLAS.axpy!(T(1), v, x)
    else
        BLAS.axpy!(T(-opt.rate), gx, x)
    end
    fill!(gx, T(0))
end

function curvature_range(g, beta)
    h = sum(map(x -> x*x, g))

end

function gradient_variance(g, beta)
    
end
