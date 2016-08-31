export Cluster, cluster

type Cluster <: Functor
    c::Var
end

function Cluster(T::Type, xsize::Int, k::Int)
    c = rand(T,xsize,k)
    Cluster(Var(c))
end

function (f::Cluster)(x::Var)
    y = cluster(x.data, f.c.data)
    df(gy) = hasgrad(x) && ∇cluster!()
    Var(y, [x], f, df)
end

function cluster{T}(x::Matrix{T}, c::Matrix{T})
    @assert size(x,1) == size(c,1)
    I, J = size(x,2), size(c,2)

    q = Array(T, I, J)
    for j = 1:J
        for i = 1:I
            q[i,j] = sqrt(1 + norm(x,i,c,j))
        end
    end
    normalize2!(q)
    f = sum(q, 1)
    p = similar(q)
    for j = 1:J
        for i = 1:I
            p[i,j] = q[i,j] * q[i,j] / f[j]
        end
    end
    normalize2!(p)
    kl_divergence(p, q)
end

function ∇cluster!{T}(p::Matrix{T}, q::Matrix, diffs::Matrix{Vector{T}}, gx::Matrix{T}, gy::Matrix{T})
    for j = 1:100
        for i = 1:100
            diff = diffs[i,j]
            (p[i,j] - q[i,j]) * sqrt(1 + norm(diff))
        end
    end
end

function Base.norm{T}(x::Matrix{T}, xj::Int, y::Matrix{T}, yj::Int)
    @assert size(x,1) == size(y,1)
    sum = T(0)
    for i = 1:size(x,1)
        d = x[i,xj] - y[i,yj]
        sum += d * d
    end
    sum
end

function normalize2!{T}(x::Matrix{T})
    for j = 1:size(x,2)
        z = T(0)
        for i = 1:size(x,1)
            z += x[i,j]
        end
        z = T(1) / z
        for i = 1:size(x,1)
            x[i,j] *= z
        end
    end
    x
end

function update!(f::Cluster, opt)
    for c in f.centroids
        opt(c.data, c.grad)
    end
end
