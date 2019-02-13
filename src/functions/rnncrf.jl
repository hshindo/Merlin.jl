export RNNCRF

mutable struct RNNCRF <: Functor
    beta::Var
    mu1::Var
    mu2::Var
    niters::Int
end

function RNNCRF(::Type{T}, insize::Int) where T
    beta = parameter(fill(T(1), insize, 1))
    mu1 = parameter(fill(T(1/insize),insize,insize))
    mu2 = parameter(fill(T(1/insize),insize,insize))
    RNNCRF(beta, mu1, mu2, 5)
end

function (nn::RNNCRF)(u::Var, dims)
    q = softmax(u)
    bos = zero(q, size(q,1), 1)
    for i = 1:nn.niters
        # q = nn.beta .* q
        q1 = nn.mu1 * q
        q2 = nn.mu2 * q
        q1s = Var[]
        q2s = Var[]
        offs = 0
        for d in dims
            if d == 1
                push!(q1s, bos)
                push!(q2s, bos)
            else
                push!(q1s, bos, q1[:,offs+1:offs+d-1])
                push!(q2s, q2[:,offs+2:offs+d], bos)
            end
            offs += d
        end
        q = concat(2, q1s...) + concat(2, q2s...)
        q += u
        i == nn.niters || (q = softmax(q))
    end
    q
end
