export crelu

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
crelu(x::Node) = Node(crelu, x)

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
    isvoid(x.grad) && return
    ∇crelu!(y.grad, x.data, x.grad)
end

function ∇crelu!(gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(x)
        k = (i-1)*2 + 1
        gx[i] += x[i] > zero(T) ? gy[k] : -gy[k+1]
    end
end
