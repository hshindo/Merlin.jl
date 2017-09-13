export crelu

doc"""
    crelu(x)

Concatenated Rectified Linear Unit.
```math
f(x) = (\max(0,x), \max(0,-x))
```
"""
crelu(x::Var) = Var(crelu(x.data), x.batchdims, crelu, (x,))

crelu(x::Node; name="crelu") = Node(crelu, x, name=name)

function crelu{T}(x::Array{T})
    s = [size(x)...]
    s[1] *= 2
    y = Array{T}(s...)
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

function ∇crelu!{T}(gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds for i = 1:length(x)
        k = (i-1)*2 + 1
        gx[i] += x[i] > T(0) ? gy[k] : -gy[k+1]
    end
end
