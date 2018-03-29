relu(x::CuArray) = CUDNN.relu(x)
sigmoid(x::CuArray) = CUDNN.sigmoid(x)
Base.tanh(x::CuArray) = CUDNN.tanh(x)

∇relu!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray) = CUDNN.∇relu!(y, gy, x, gx)
∇sigmoid!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray) = CUDNN.∇sigmoid!(y, gy, x, gx)
∇tanh!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray) = CUDNN.∇tanh!(y, gy, x, gx)
