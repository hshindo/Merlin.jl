export elu

doc"""
    elu(x)

Exponential Linear Unit.
```math
f(x) =
\begin{cases}
x & x > 0 \\
\alpha (e^{x}-1) & x\leq0
\end{cases}
```
"""
elu(x::Var) = Var(elu(x.data), elu, (x,))

elu(x::Node) = Node(elu, x)

function elu{T}(x::Array{T})
    alpha = T(1)
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = x[i] > 0 ? x[i] : alpha*T(exp(x[i])-1)
    end
    y
end

function addgrad!(y::Var, ::typeof(elu), x::Var)
    isvoid(x.grad) && return
    ∇elu!(y.data, y.grad, x.data, x.grad)
end

function ∇elu!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    alpha = T(1)
    @inbounds for i = 1:length(x)
        gx[i] += x[i] > T(0) ? gy[i] : gy[i]*(y[i]+alpha)
    end
end
