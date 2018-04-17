export leaky_relu

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
