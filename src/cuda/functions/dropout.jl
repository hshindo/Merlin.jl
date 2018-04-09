dropout(x::CuArray, droprate) = CUDNN.dropout(x, droprate)

∇dropout!(gy::CuArray, gx, droprate, r) = CUDNN.∇dropout!(gy, gx, droprate, r)
