export MLP

mutable struct MLP
    W
    b
    σ
end

doc"""
    MLP

Multi-layer perceptron
"""
function MLP(::Type{T}, insize::Int, outsize::Int, activation=identity; init_W, init_b) where T

end

function (f::MLP)(x)
    y = σ(f.W * x .+ f.b)
end
