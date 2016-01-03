function checkstatus(status)
  if status != STATUS_SUCCESS
      str = bytestring(ccall((:cudnnGetErrorString,libcudnn), Cstring, (Cint,), status))
      throw(str)
  end
end

function cudnnCreate(handle)
  ccall((:cudnnCreate,libcudnn), Cint, (Ptr{Ptr{Void}},), handle) |> checkstatus
end

function cudnnDestroy(handle)
  ccall(:cudnnDestroy, (Ptr{Void},), handle)
end

function set_stream(handle, stream)
  @cudnncall(:cudnnSetStream, (Ptr{Void},Ptr{Void}), handle, stream)
end

function get_stream(handle)
  s_handle = Ptr{Void}[0]
  @cudnncall(:cudnnGetStream, (Ptr{Void},Ptr{Ptr{Void}}), handle, s_handle)
  return s_handle[1]
end
