export SGD

"""
    SGD

Stochastic Gradient Descent.

## Arguments
* rate: learning rate
* [momentum::Float64]: momentum coefficient
* [nesterov::Bool]: us nesterov acceleration or not
"""
type SGD
    rate::Float64
    momentum::Float64
    nesterov::Bool
    states::ObjectIdDict
end

function SGD(rate=0.0; momentum=0.0, nesterov=false)
    SGD(rate, momentum, nesterov, ObjectIdDict())
end

function (opt::SGD){T,N}(x::Array{T,N}, gx::Array{T,N})
    if opt.momentum > 0.0
        v = get!(()->zeros(x), opt.states, x)
        BLAS.scal!(length(v), T(opt.momentum), v, 1)
        BLAS.axpy!(T(-opt.rate), gx, v)
        if nesterov
            BLAS.scal!(length(v), T(opt.momentum), v, 1)
            BLAS.axpy!(T(-opt.rate), gx, v)
        end
        add!(x, v)
    else
        BLAS.axpy!(T(-opt.rate), gx, x)
    end
    fill!(gx, T(0))
end
