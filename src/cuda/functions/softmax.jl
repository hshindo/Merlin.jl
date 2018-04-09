softmax(x::CuArray) = CUDNN.softmax(x)

∇softmax!(y::CuArray, gy::CuArray, gx::CuArray) = CUDNN.∇softmax!(y, gy, gx)

logsoftmax(x::CuArray) = CUDNN.logsoftmax(x)

∇logsoftmax!(y::CuArray, gy::CuArray, gx::CuArray) = CUDNN.∇logsoftmax!(y, gy, gx)
