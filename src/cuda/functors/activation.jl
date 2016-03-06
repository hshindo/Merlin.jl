type Activation <: Functor
  mode::ASCIIString
end

function relu{T,N}(x::CudaArray{T,N})
  y = alloc(T, size(x))
  activation_forward!(CUDNN_ACTIVATION_RELU, 1.0, x, 0.0, y)
  y
end

#function getmode(mode)
  #  mode == "sigmoid" && return CUDNN_ACTIVATION_SIGMOID
  #  mode == "relu" && return CUDNN_ACTIVATION_RELU
  #  mode == "tanh" && return CUDNN_ACTIVATION_TANH
  #  mode == "clipped_relu" && return CUDNN_ACTIVATION_CLIPPED_RELU
  #  throw("Invalid activation mode: $mode")
  #end
