function softmax(x::CuArray)
    CUDNN.softmax(x)
end

function ∇softmax!(y::CuArray, gy::CuArray, gx::CuArray)
    CUDNN.∇softmax!(y, gy, gx)
end

function logsoftmax(x::CuArray)
    CUDNN.logsoftmax(x)
end

function ∇logsoftmax!(y::CuArray, gy::CuArray, gx::CuArray)
    CUDNN.∇logsoftmax!(y, gy, gx)
end
