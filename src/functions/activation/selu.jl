export selu

doc"""
    selu(x::Var)

Scaled Exponential Linear Unit.

```math
f(x) = \lambda
\begin{cases}
x & x > 0 \\
\alpha e^{x}-\alpha & x\leq0
\end{cases}
```
where ``\lambda=1.0507`` and ``\alpha=1.6733``.

# References
Klambauer et al., ["Self-Normalizing Neural Networks"](https://arxiv.org/abs/1706.02515), NIPS 2017.
"""
selu(x::Var) = Var(selu.(x.data), (selu,x))
selu(x::T) where T = x > 0 ? T(1.0507)*x : T(1.0507)*T(1.6733)*(exp(x)-1)
selu(x::Node) = Node(selu, x)

function addgrad!(y::Var, ::typeof(selu), x::Var)
    isvoid(x.grad) || ∇selu!(y.data, y.grad, x.data, x.grad)
end

function ∇selu!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    alpha = T(1.6733)
    lambda = T(1.0507)
    @inbounds for i = 1:length(x)
        gx[i] +=  x[i] > 0 ? gy[i]*lambda : gy[i]*(y[i]+lambda*alpha)
    end
end
