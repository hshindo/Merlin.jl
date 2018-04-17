export swish

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
