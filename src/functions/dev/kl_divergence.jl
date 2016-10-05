export kl_divergence

function kl_divergence(p::Var, q::Var)
    y = kl_divergence(p.data, q.data)
    function df(gy)
        hasgrad(p) && ∇kl_divergence_p!(p.data, p.grad, q.data)
        hasgrad(q) && ∇kl_divergence_q!(p.data, q.data, q.grad)
    end
    Var(y, [p,q], kl_divergence, df)
end

function kl_divergence{T}(p::Array{T}, q::Array{T})
    d = T(0)
    @inbounds @simd for i = 1:length(p)
        d += p[i] * (log(p[i]) - log(q[i]))
    end
    d
end

function ∇kl_divergence_p!{T}(p::Array{T}, gp::Array{T}, q::Array{T})
    @inbounds @simd for i = 1:length(p)
        gp[i] += log(p[i]) - log(q[i]) + 1
    end
end

function ∇kl_divergence_q!{T}(p::Array{T}, q::Array{T}, gq::Array{T})
    @inbounds @simd for i = 1:length(p)
        gq[i] += -p[i] / q[i]
    end
end
