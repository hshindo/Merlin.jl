relu(x::CuArray) = CUDNN.relu(x)

∇relu!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray) = CUDNN.∇relu!(y, gy, x, gx)

sigmoid(x::CuArray) = CUDNN.sigmoid(x)

∇sigmoid!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray) = CUDNN.∇sigmoid!(y, gy, x, gx)

Base.tanh(x::CuArray) = CUDNN.tanh(x)

∇tanh!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray) = CUDNN.∇tanh!(y, gy, x, gx)
