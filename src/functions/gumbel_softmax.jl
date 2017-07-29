export gumbel_softmax

"""
    gumbel_softmax(x::Var, tau::Float64)

* x: logarithm of probability distribution, e.g., output of `logsoftmax`.
* tau: temprature
"""
function gumbel_softmax(x::Var, tau::Float64)
    tau > 0.0 || throw("tau: $(tau) must be positive.")
    y = Var(nothing, gumbel_softmax, (x,tau))
    y.data = gumbel_softmax(x.data)
    y.df! = () -> begin
        isvoid(x.grad) && return
        ∇gumbel_softmax!(y.data, y.grad, x.grad, eltype(x.data)(tau))
    end
    y
end

gumbel{T}(u::T) = -log(-log(u+1e-20)+1e-20)

function gumbel_softmax_train{T}(logp::Matrix{T})
    y = Array{Int}(size(logp,2))
    u = rand(T, size(logp))
    @inbounds for j = 1:size(logp,2)
        maxi, maxv = 0, -1e100
        for i = 1:size(logp,1)
            v = logp[i,j] + gumbel(u[i,j])
            if v > maxv
                maxi = i
                maxv = v
            end
        end
        y[j] = maxi
    end
    y
end

function gumbel_softmax_unnormalized{T}(x::Matrix{T}, tau::T)
    y = similar(x)
    u = rand(T, size(x))
    tau = T(1) / tau
    @inbounds for i = 1:length(x)
        y[i] = (x[i] + gumbel(u[i])) * tau
    end
    y
end

function ∇gumbel_softmax!{T}(y::Array{Int}, gy::Array{T}, gx::Array{T}, tau::T)
    for j = 1:size(x,2)
        maxv = x[1,j]
        for i = 1:size(x,1)
            maxv = max(maxv, x[i,j])
        end

        z = T(0)
        for i = 1:size(x,1)
            y = exp(x[i,j] - maxv)
            z += y
        end
        z == T(0) && error("z == 0")
        invz = 1 / z
        for i = 1:size(x,1)
            y[i,j] *= invz
        end
    end

    for j = 1:size(y,2)
        sum = T(0)
        for i = 1:size(y,1)
            sum += gy[i,j] * y[i,j]
        end
        for i = 1:size(y,1)
            gx[i,j] += y[i,j] * (gy[i,j]-gy[k,j])
        end
    end

    for j = 1:size(gx,2)
        k = y[j]
        g = gy[k,j]
        for i = 1:size(y,1)
            i == k && continue
            delta = ifelse(i==k, T(0), -g)
            gx[i,j] += y[i,j] * delta / tau
        end

    end
end
