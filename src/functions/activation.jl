export crelu, elu, leaky_relu, relu, selu, sigmoid, swish
import Base.tanh

doc"""
    crelu(x)

Concatenated Rectified Linear Unit.
The output is twice the size of the input.

```math
f(x) = (\max(0,x), \max(0,-x))
```
"""
function crelu(x::Var)
    Var(crelu(x.data), x.batchdims, crelu, (x,))
end

crelu(x::Node; name="") = Node(crelu, (x,), name)

function crelu(x::Array{T}) where T
    y = Array{T}(2size(x,1), Base.tail(size(x))...)
    @inbounds for i = 1:length(x)
        k = (i-1)*2 + 1
        y[k] = max(x[i], T(0))
        y[k+1] = max(-x[i], T(0))
    end
    y
end

function addgrad!(y::Var, ::typeof(crelu), x::Var)
    isvoid(x.grad) || ∇crelu!(y.grad, x.data, x.grad)
end

function ∇crelu!(gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(x)
        k = (i-1)*2 + 1
        gx[i] += x[i] > T(0) ? gy[k] : -gy[k+1]
    end
end

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
elu(x::Var) = Var(elu(x.data), x.batchdims, elu, (x,))

elu(x::Node; name="") = Node(elu, (x,), name)

function elu(x::Array{T}) where T
    alpha = T(1)
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = x[i] > 0 ? x[i] : alpha*T(exp(x[i])-1)
    end
    y
end

function addgrad!(y::Var, ::typeof(elu), x::Var)
    isvoid(x.grad) || ∇elu!(y.data, y.grad, x.data, x.grad)
end

function ∇elu!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    alpha = T(1)
    @inbounds for i = 1:length(x)
        gx[i] += x[i] > T(0) ? gy[i] : gy[i]*(y[i]+alpha)
    end
end

doc"""
    sigmoid(x)

Sigmoid logistic function.

```math
f(x) = (1 + \exp(-x))^{-1}
```
"""
sigmoid(x::Var) = Var(sigmoid.(x.data), x.batchdims, sigmoid, (x,))

sigmoid(x::Node; name="") = Node(sigmoid, (x,), name)

sigmoid(x::T) where T<:AbstractFloat = 1 / (1 + exp(-x))

function addgrad!(y::Var, ::typeof(sigmoid), x::Var)
    isvoid(x.grad) || ∇sigmoid!(y.data, y.grad, x.data, x.grad)
end

function ∇sigmoid!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end

doc"""
    tanh(x)

Hyperbolic tangent function.
"""
tanh(x::Var) = Var(tanh.(x.data), x.batchdims, tanh, (x,))

tanh(x::Node; name="") = Node(tanh, (x,), name)

function addgrad!(y::Var, ::typeof(tanh), x::Var)
    isvoid(x.grad) || ∇tanh!(y.data, y.grad, x.data, x.grad)
end

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end
