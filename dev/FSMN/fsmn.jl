export FSMN

"""
    FSMN

Feedforward Sequential Memory Networks.

Ref: https://arxiv.org/abs/1512.08301
"""
function FSMN(T::Type, hsize::Int)
    embed = Embedding(T, 30000, hsize)
    ls = [Linear(T,hsize), Linear(T)]
    @graph begin
        h1 = embed(:x)
        a = attention(:a)
        h2 = gemm(h1, a)
        x = ls[1](h1) + ls[2](h2)
        x = relu(x)
        x
    end
end

function attentionmat(x::Var)
end

function upperband{T}(a::Vector{T}, n::Int)
    M = zeros(T, n, n)
    for i = 1:n
        len = min(length(a), n-i+1)
        copy!(M, (i-1)*n+i, a, 1, len)
    end
    M
end
