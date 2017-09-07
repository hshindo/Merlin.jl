export selu

doc"""
    selu(x)

Scaled Exponential Linear Unit.
```math
f(x) = \lambda
\begin{cases}
x & x > 0 \\
\alpha e^{x}-\alpha & x\leq0
\end{cases}
```

Reference: Klambauer et al., [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515), NIPS 2017
"""
selu(x::Var) = Var(selu(x.data), selu, (x,))

selu(x::Node) = Node(selu, x)

function selu{T}(x::Array{T})
    alpha = T(1.6733)
    lambda = T(1.0507)
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = x[i] > 0 ? lambda*x[i] : lambda*alpha*T(exp(x[i]) - 1)
    end
    y
end

function addgrad!(y::Var, ::typeof(selu), x::Var)
    isvoid(x.grad) || ∇selu!(y.data, y.grad, x.data, x.grad)
end

function ∇selu!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    alpha = T(1.6733)
    lambda = T(1.0507)
    @inbounds for i = 1:length(x)
        gx[i] +=  x[i] > 0 ? gy[i]*lambda : gy[i]*(y[i]+lambda*alpha)
    end
end
