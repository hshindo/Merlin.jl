export crossentropy

"""
    crossentropy(p,x)

Computes cross-entropy between p and x. x is assumed to be unnormalized.

* p: Vector{Int} or Matrix{Float}

## 👉 Example
```julia
p = [1:5;]
x = Var(rand(Float32,10,5))
y = crossentropy(p,x)
```
"""
function crossentropy(p::Vector{Int}, x::Var)
    x.data == nothing && return Var(nothing, (crossentropy,p,x))
    logx = alloc(eltype(x), size(x))
    logsoftmax!(x.data, logx)
    y = Var(eltype(x), (1,length(p)), (x,))
    crossentropy!(p, logx, y.data)
    y.df = () -> isconst(x) || ∇crossentropy!(y.grad, p, logx, x.grad)
    y

    #logx = logsoftmax(x.data)
    #y = crossentropy(p, logx)
    #df(gy) = isconst(x) || ∇crossentropy!(p, logx, x.grad, gy)
    #Var(y, [x], crossentropy, df)
end

function crossentropy{T}(p::Matrix{T}, logx::Matrix{T})
    y = Array(T, 1, size(p,2))
    for j = 1:size(p,2)
        s = T(0)
        @inbounds @simd for i = 1:size(p,1)
            s += -p[i,j] * logx[i,j]
        end
        y[j] = s
    end
    y
end

function crossentropy!{T}(p::Vector{Int}, logx::Matrix{T}, y::Matrix{T})
    @inbounds @simd for j = 1:length(p)
        y[j] = -logx[p[j],j]
    end
end

function ∇crossentropy2!{T}(p::Matrix{T}, logx::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
    for j = 1:size(p,2)
        g = gy[j]
        @inbounds @simd for i = 1:size(p,1)
            gx[i,j] += g * (exp(logx[i,j]) - p[i,j])
        end
    end
end

function ∇crossentropy!{T}(gy::Matrix{T}, p::Vector{Int}, logx::Matrix{T}, gx::Matrix{T})
    for j = 1:length(p)
        g = gy[j]
        @inbounds @simd for i = 1:size(logx,1)
            delta = ifelse(i == p[j], T(1), T(0))
            gx[i,j] += g * (exp(logx[i,j]) - delta)
        end
    end
end
