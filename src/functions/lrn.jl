export lrn
lrn(x::CuArray) = JuCUDNN.lrn(x)
∇lrn!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray) = JuCUDNN.∇lrn!(x, y, gy, gx)
