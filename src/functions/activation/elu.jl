export elu

doc"""
    elu(x::Var)

Exponential Linear Unit.

# References
* Clevert et al., ["Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"](https://arxiv.org/abs/1511.07289), arXiv 2015.

```math
f(x) =
\begin{cases}
x & x > 0 \\
\alpha (e^{x}-1) & x\leq0
\end{cases}
```
where ``\alpha=1``.
"""
elu(x::Var) = Var(elu.(x.data), (elu,x))
elu(x::T) where T = x > zero(T) ? x : exp(x)-1
elu(x::Node) = Node(elu, x)

function addgrad!(y::Var, ::typeof(elu), x::Var)
    isvoid(x.grad) && return
    ∇elu!(y.data, y.grad, x.data, x.grad)
end

function ∇elu!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    alpha = one(T)
    @inbounds for i = 1:length(x)
        gx[i] += x[i] > zero(T) ? gy[i] : gy[i]*(y[i]+alpha)
    end
end
