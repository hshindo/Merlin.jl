export mse

doc"""
    mse(x1, x2)

Mean Squared Error function between `x1` and `x2`.
The mean is calculated over the minibatch.
Note that the error is not scaled by 1/2.
"""
function mse(x1::Var, x2::Var)
    Var(mse(x1.data,x2.data), (mse,x1,x2))
end

function mse(x1::Matrix{T}, x2::Matrix{T}) where T
    size(x1) == size(x2) || throw("Size unmatch.")
    y = similar(x1, size(x1,2))
    for j = 1:size(x1,2)
        v = T(0)
        for i = 1:size(x1,1)
            d = x1[i,j] - x2[i,j]
            v += d * d
        end
        y[j] = v / size(x1,1)
    end
    y
end

function addgrad!(y::Var, ::typeof(mse), x1::Var, x2::Var)
    T = eltype(y)
    gx1 = isvoid(x1.grad) ? Array{T}(0,0) : x1.grad
    gx2 = isvoid(x2.grad) ? Array{T}(0,0) : x2.grad
    ∇mse!(y.grad, x1.data, gx1, x2.data, gx2)
end

function ∇mse!(gy::Vector{T}, x1::Matrix{T}, gx1::Matrix{T}, x2::Matrix{T}, gx2::Matrix{T}) where T
    for j = 1:size(x1,2)
        for i = 1:size(x1,1)
            g = gy[j] * (x1[i,j]-x2[i,j]) * 2 / size(x1,1)
            isempty(gx1) || (gx1[i,j] += g)
            isempty(gx2) || (gx2[i,j] -= g)
        end
    end
end
