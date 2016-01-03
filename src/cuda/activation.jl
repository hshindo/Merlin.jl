module Activation

const SIGMOID = 0
const RELU = 1
const TANH = 2
const CLIPPED_RELU = 3

function activation_forward(mode::Int, x::AbstractCudaArray, y::AbstractCudaArray; xdesc, ydesc, alpha=1.0, beta=0.0)
  handle = gethandle(x.dev)
  @cudnncall(:cudnnActivationForward, (Ptr{Void},Cint,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
    handle, mode, [alpha], xdesc, x, [beta], ydesc, y)
end

end
