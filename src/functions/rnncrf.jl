export RNNCRF

mutable struct RNNCRF <: Functor
    beta::Var
    cnn
    mu::Var
end

function RNNCRF(::Type{T}, insize::Int, temp) where T
    beta = Var(fill(T(-1/temp), insize))
    cnn = Conv1d(T, 3, insize, insize, padding=1)
    mu = fill(T(1/insize), insize, insize)
    RNNCRF(cnn, parameter(mu))
end

function (nn::RNNCRF)(u::Var, x::Var, dims, niters::Int)
    

    q = nn.beta .* u
    q = softmax(q)
    for i = 1:niters
        # q = concat(1, q, f)
        q = nn.cnn(q, dims)
        q = nn.mu * q
        q += u
        i == niters || (q = softmax(q))
    end
    q
end
