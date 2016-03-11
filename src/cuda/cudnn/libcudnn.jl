function cudnnCheck(status)
  if status != STATUS_SUCCESS
      str = bytestring(ccall((:cudnnGetErrorString,libcudnn), Cstring, (Cint,), status))
      throw(str)
  end
end

function cudnnCreate(handle)
  ccall((:cudnnCreate,libcudnn), Cint, (Ptr{Ptr{Void}},), handle) |> cudnnCheck
end

function cudnnDestroy(handle)
  ccall((:cudnnDestroy,libcudnn), Cint, (Ptr{Void},), handle) |> cudnnCheck
end

function cudnnDestroyTensorDescriptor(tensorDesc)
  ccall((:cudnnDestroyTensorDescriptor,libcudnn), Cint, (Ptr{Void},), tensorDesc) |> cudnnCheck
end

function cudnnCreateTensorDescriptor(tensorDesc)
  ccall((:cudnnCreateTensorDescriptor,libcudnn), Cint, (Ptr{Void},), tensorDesc) |> cudnnCheck
end

function cudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA)
  ccall((:cudnnSetTensorNdDescriptor,libcudnn), Cint, (Ptr{Void},Cint,Cint,Ptr{Cint},Ptr{Cint}), tensorDesc, dataType, nbDims, dimA, strideA) |> cudnnCheck
end

function cudnnActivationForward(handle, mode, alpha, xDesc, x, beta, yDesc, y)
  ccall((:cudnnActivationForward,libcudnn), Cint, (Ptr{Void},Cint,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
    handle, mode, alpha, xDesc, x, beta, yDesc, y) |> cudnnCheck
end

function cudnnActivationBackward(handle, mode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
  ccall((:cudnnActivationBackward,libcudnn), Cint, (Ptr{Void},Cint,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
    handle, mode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx) |> cudnnCheck
end
