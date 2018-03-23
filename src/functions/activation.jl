export crelu, elu, leaky_relu, relu, selu, sigmoid, swish

doc"""
    crelu(x::Var)

Concatenated Rectified Linear Unit.
The output is twice the size of the input.

```math
f(x) = (\max(0,x), \max(0,-x))
```

# References
* Shang et al., ["Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units"](https://arxiv.org/abs/1603.05201), arXiv 2016.
"""
crelu(x::Var) = Var(crelu(x.data), (crelu,x))

function crelu(x::Array{T}) where T
    y = Array{T}(2size(x,1), Base.tail(size(x))...)
    @inbounds for i = 1:length(x)
        k = (i-1)*2 + 1
        y[k] = max(x[i], zero(T))
        y[k+1] = max(-x[i], zero(T))
    end
    y
end

function addgrad!(y::Var, ::typeof(crelu), x::Var)
    isvoid(x.grad) || ∇crelu!(y.grad, x.data, x.grad)
end

function ∇crelu!(gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(x)
        k = (i-1)*2 + 1
        gx[i] += x[i] > zero(T) ? gy[k] : -gy[k+1]
    end
end

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

doc"""
    leaky_relu(x::Var, alpha=0.1)

Leaky Rectified Linear Unit.

```math
f(x) =
\begin{cases}
x & x > 0 \\
\alpha x & x \leq 0
\end{cases}
```

# References
* Maas et al., ["Rectifier Nonlinearities Improve Neural Network Acoustic Models"](http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf), ICML 2013.
"""
leaky_relu(x::Var, alpha=0.1) = Var(leaky_relu.(x.data,eltype(x)(alpha)), (leaky_relu,x,alpha))
leaky_relu(x::T, alpha::T) where T = x >= zero(T) ? x : x*alpha
leaky_relu(x::Node, alpha=0.1) = Node(leaky_relu, x, alpha)

function addgrad!(y::Var, ::typeof(leaky_relu), x::Var, alpha::Float64)
    isvoid(x.grad) && return
    ∇leaky_relu!(y.grad, x.data, x.grad, eltype(x)(alpha))
end

function ∇leaky_relu!(gy::Array{T}, x::Array{T}, gx::Array{T}, alpha::T) where T
    @inbounds for i = 1:length(x)
        gx[i] += x[i] >= zero(T) ? gy[i] : gy[i]*alpha
    end
end

doc"""
    relu(x::Var)

Rectified Linear Unit.

```math
f(x) = \max(0, x)
```
"""
relu(x::Var) = Var(relu(x.data), (relu,x))
relu(x::T) where T<:AbstractFloat = max(x, zero(T))
relu(x::Array) = relu.(x)
relu(x::Node) = Node(relu, x)

function addgrad!(y::Var, ::typeof(relu), x::Var)
    isvoid(x.grad) && return
    ∇relu!(y.data, y.grad, x.data, x.grad)
end

function ∇relu!(y, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(x)
        gx[i] += x[i] > T(0) ? gy[i] : T(0)
    end
end

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

doc"""
    sigmoid(x)

Sigmoid logistic function.

```math
f(x) = (1 + \exp(-x))^{-1}
```
"""
sigmoid(x::Var) = Var(sigmoid(x.data), (sigmoid,x))
sigmoid(x::Array) = sigmoid.(x)
sigmoid(x::T) where T<:AbstractFloat = T(1 / (1 + exp(-x)))
sigmoid(x::Node) = Node(sigmoid, x)

function addgrad!(y::Var, ::typeof(sigmoid), x::Var)
    isvoid(x.grad) && return
    ∇sigmoid!(y.data, y.grad, x.data, x.grad)
end

function ∇sigmoid!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end

doc"""
    Swish

Swish activation function.

```math
f(x) = x \cdot \sigma (\beta x)
```
where ``\beta`` is a leanable parameter.

# References
* Ramachandran et al. ["Searching for Activation Functions"](https://arxiv.org/abs/1710.05941), arXiv 2017.
"""
struct Swish
    beta::Var
end
Swish(::Type{T}) where T = Swish(zerograd(ones(T,1)))
(f::Swish)(x) = swish(x, f.beta)

swish(x::Var, beta::Var) = Var(swish.(x.data,beta.data), (swish,x,beta))
swish(x::T, beta::T) where T = x * sigmoid(beta*x)

function addgrad!(y::Var, ::typeof(swish), x::Var, beta::Var)
    isvoid(x.grad) || ∇swish_x!(y.data, y.grad, x.data, x.grad, beta.data)
    isvoid(beta.grad) || ∇swish_beta!(y.data, y.grad, x.data, beta.data, beta.grad)
end

function ∇swish_x!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}, beta::Vector{T}) where T
    @inbounds for i = 1:length(x)
        s = sigmoid(beta[1] * x[i])
        gx[i] += gy[i] * (s + beta[1]*x[i] * s * (1-s))
    end
end

function ∇swish_beta!(y::Array{T}, gy::Array{T}, x::Array{T}, beta::Vector{T}, gbeta::Vector{T}) where T
    @inbounds for i = 1:length(gy)
        s = sigmoid(beta[1] * x[i])
        gbeta[1] += gy[i] * x[i] * x[i] * s * (1-s)
    end
end

doc"""
    tanh(x::Var)

Hyperbolic tangent function.
"""
Base.tanh(x::Var) = Var(tanh(x.data), (tanh,x))
Base.tanh(x::Array) = tanh.(x)
Base.tanh(x::Node) = Node(tanh, x)

function addgrad!(y::Var, ::typeof(tanh), x::Var)
    isvoid(x.grad) && return
    ∇tanh!(y.data, y.grad, x.data, x.grad)
end

function ∇tanh!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end
