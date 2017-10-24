export logsoftmax

"""
    logsoftmax(x)

Logarithm of softmax function.
"""
logsoftmax(x::Var) = Var(logsoftmax(x.data), x.batchdims, logsoftmax, (x,))

logsoftmax(x::Node; name="") = Node(logsoftmax, (x,), name)

function logsoftmax{T}(x::Matrix{T})
    y = similar(x)
    max = maximum(x, 1)
    @inbounds for j = 1:size(x,2)
        sum = T(1e-10)
        for i = 1:size(x,1)
            sum += exp(x[i,j] - max[j])
        end
        logz = log(sum)
        for i = 1:size(x,1)
            y[i,j] = x[i,j] - max[j] - logz
        end
    end
    y
end

function addgrad!(y::Var, ::typeof(logsoftmax), x::Var)
    isvoid(x.grad) || ∇logsoftmax!(y.data, y.grad, x.grad)
end

function ∇logsoftmax!{T}(y::Matrix{T}, gy::Matrix{T}, gx::Matrix{T})
    @inbounds for j = 1:size(y,2)
        sum = T(0)
        for i = 1:size(y,1)
            sum += gy[i,j]
        end
        for i = 1:size(y,1)
            gx[i,j] += gy[i,j] - exp(y[i,j]) * sum
        end
    end
end
