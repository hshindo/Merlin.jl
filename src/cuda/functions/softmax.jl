function softmax!(out, x::CuArray{T,N}) where {T,N}
    throw("Not implemented yet.")
end

softmax(x::CuArray) = CUDNN.softmax(x)
logsoftmax(x::CuArray) = CUDNN.logsoftmax(x)
