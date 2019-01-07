export RNNCRF

mutable struct RNNCRF <: Functor
    mu::Var
    niters::Int
end

function RNNCRF(::Type{T}, insize::Int, niters::Int) where T
    mu = fill(T(1/insize), insize, insize)
    mu = parameter(mu)
    RNNCRF(mu, niters)
end

function (f::RNNCRF)(x::Var)
    q = softmax(x)
    for i = 1:f.niters
        q = f.μ * q
        q += x
        q = softmax(q)
    end
    q
end
