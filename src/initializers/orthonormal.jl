export OrthoNormal

doc"""
    OrthoNormal
"""
struct OrthoNormal
end

function (o::OrthoNormal)(::Type{T}, dim1::Int, dim2::Int) where T
    I = eye(dim2)
    lr = 0.1
    eps = 0.05 / (dim1+dim2)
    tries = 0
    while tries < 10
        Q = randn(dim1, dim2) / sqrt(dim2)
        for i = 1:100
            QTQmI = Q' * Q - I
            loss = sum(QTQmI .^ 2 / 2)
            Q2 = Q .^ 2
            a = abs.(Q2 .+ sum(Q2,1) .+ sum(Q2,2) - 1.0) + eps
            Q -= lr * Q * QTQmI ./ a
            if maximum(Q) > 1e6 || loss > 1e6 || isinf(loss)
                tries += 1
                lr /= 2.0
                break
            end
        end
        return Matrix{T}(Q)
    end
    #Q = xp.random.randn(input_size, output_size) / xp.sqrt(output_size)
    throw("Generation failed.")
end
