export leaky_relu

doc"""
    leaky_relu(x, alpha=10)

Leaky Rectified Linear Unit.
```math
f(x) =
\begin{cases}
x & x > 0 \\
x/\alpha & x \leq 0
\end{cases}
```
"""
leaky_relu(x::Var, alpha=10) = Var(leaky_relu(x.data,alpha), x.batchdims, leaky_relu, (x,alpha))

leaky_relu(x::Node; name="leaky_relu") = Node(leaky_relu, x, name=name)

function leaky_relu{T}(x::Array{T}, alpha::Number)
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = x[i] > T(0) ? x[i] : x[i]/T(alpha)
    end
    y
end

function addgrad!(y::Var, ::typeof(leaky_relu), x::Var, alpha::Number)
    isvoid(x.grad) || ∇leaky_relu!(y.grad, x.data, x.grad, alpha)
end

function ∇leaky_relu!{T}(gy::Array{T}, x::Array{T}, gx::Array{T}, alpha::Number)
    @inbounds for i = 1:length(x)
        gx[i] += x[i] > T(0) ? gy[i] : gy[i]/T(alpha)
    end
end
