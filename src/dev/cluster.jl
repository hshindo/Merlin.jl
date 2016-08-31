type Cluster <: Functor
    mus::Vector
end

function (f::Cluster)(x::Var)
    y = p(x.data)
    df(gy) = hasgrad(x) && ∇cluster!()
    Var(y, [x], f, df)
end

function cluster{T}(f::Cluster, x::Array{T})
    q = map(f.mus) do mu
        d = x[i] - mu
        sqrt(1 + d * d)
    end
    normalize!(q, 1)

    for i = 1:length(f.mus)
        f.mus[i]
    end
end

function ∇cluster!()
end
