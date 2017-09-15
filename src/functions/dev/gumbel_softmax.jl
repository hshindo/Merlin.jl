export gumbel_softmax

"""
    gumbel_softmax(x::Var, tau)

* x: logarithm of probability distribution, e.g., output of `logsoftmax`.
* tau: temprature
"""
function gumbel_softmax(x::Var, tau)
    tau > 0.0 || throw("tau: $(tau) must be positive.")
    tau = eltype(x.data)(tau)
    y = Var(nothing, gumbel_softmax, (x,tau))
    g = gumbel_softmax(x.data, tau)
    y.data = vec(argmax(g,1))
    y.grad = zeros(g)
    y.df! = () -> begin
        isvoid(x.grad) && return
        ∇gumbel_softmax!(g, y.grad, x.grad, tau)
    end
    y
end

gumbel{T}(u::T) = -log(-log(u+1e-20)+1e-20)

function gumbel_softmax{T}(x::Matrix{T}, tau::T)
    y = similar(x)
    u = rand(T, size(x))
    tau = T(1) / tau
    @inbounds for i = 1:length(x)
        y[i] = (x[i] + gumbel(u[i])) * tau
    end
    softmax(y)
end

function ∇gumbel_softmax!{T}(y::Vector{Int}, gy::Matrix{T}, work::Matrix{T}, gx::Matrix{T}, tau::T)
    for j = 1:size(y,2)
        sum = T(0)
        for i = 1:size(y,1)
            sum += gy[i,j] * y[i,j]
        end
        y[j]
        for i = 1:size(y,1)
            gx[i,j] += y[i,j] * (gy[i,j] - sum)
        end
    end
end
