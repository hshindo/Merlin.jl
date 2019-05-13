export SGD

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
    weight_decay::Float64
    states::IdDict
end

function SGD(rate=0.0; momentum=0.0, nesterov=false, weight_decay=0.0)
    SGD(rate, momentum, nesterov, weight_decay, IdDict())
end

(opt::SGD)(x::Var) = opt(x.data, x.grad)

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
        axpy!(T(1), v, x)
    else
        if opt.weight_decay > 0.0
            axpy!(T(opt.weight_decay), x, gx)
        end
        axpy!(T(-opt.rate), gx, x)
    end
    fill!(gx, T(0))
end

function (opt::SGD)(x::CuArray{T,N}, gx::CuArray{T,N}) where {T,N}
    if opt.momentum > 0.0
        throw("Not implemented yet.")
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
            scal!(length(v), m, v, 1)
            axpy!(-rate, gx, v)
        end
        axpy!(T(1), v, x)
    else
        if opt.weight_decay > 0.0
            axpy!(T(opt.weight_decay), x, gx)
        end
        axpy!(T(-opt.rate), gx, x)
    end
    fill!(gx, T(0))
end
