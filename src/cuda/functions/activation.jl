function relu(x::CuArray)
    CUDNN.relu(x)
end

function ∇relu!(y::CuArray, gy, x, gx)
    CUDNN.∇relu!(y, gy, x, gx)
end

function sigmoid(x::CuArray)
    CUDNN.sigmoid(x)
end

function ∇sigmoid!(y::CuArray, gy, x, gx)
    CUDNN.∇sigmoid!(y, gy, x, gx)
end

function Base.tanh(x::CuArray)
    CUDNN.tanh(x)
end

function ∇tanh!(y::CuArray, gy, x, gx)
    CUDNN.∇tanh!(y, gy, x, gx)
end
