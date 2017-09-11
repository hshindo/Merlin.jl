export mse

doc"""
    mse(x1, x2)

Mean Squared Error function between `x1` and `x2`.
The mean is calculated over the minibatch.
Note that the error is not scaled by 1/2.
"""
function mse(x1::Var, x2::Var)
    x1.batchdims == x2.batchdims || throw("Batchdims mismatch: $(x1.batchdims) : $(x2.batchdims)")
    Var(mse(x1.data,x2.data), x1.batchdims, mse, (x1,x2))
end

mse(x1::Node, x2::Node) = Node(mse, x1, x2)

function mse{T}(x1::Matrix{T}, x2::Matrix{T})
    size(x1) == size(x2) || throw("Size unmatch.")
    res = similar(x1, size(x1,2))
    @inbounds for j = 1:size(x1,2)
        v = T(0)
        for i = 1:size(x1,1)
            diff = x2[i,j] - x1[i,j]
            v += diff * diff
        end
        res[j] = v / size(x1,1)
    end
    res
end

function addgrad!(y::Var, ::typeof(mse), x1::Var, x2::Var)
    isvoid(x2.grad) || ∇mse!(y.grad, x1.data, x1.grad, x2.data)
end

function ∇mse!{T}(gy::Vector{T}, x1::Matrix{T}, gx1::Matrix{T}, x2::Matrix{T})
    @inbounds for j = 1:size(x1,2)
        for i = 1:size(x1,1)
            g = gy[j] * (x2[i,j] - x1[i,j]) * 2 / size(x1,1)
            gx1[i,j] -= g
            #gx2[i] += g
        end
    end
end
