module CUDA

include("libcuda-7.5.jl")
include("libcuda-7.5_h.jl")

function check_error(status)
  if status != cudaSuccess
    warn("CUDA error triggered from:")
    Base.show_backtrace(STDOUT, backtrace())
    bytestring(cudaGetErrorString(status)) |> throw
  end
end

function check_nvrtc(status)
  if status != NVRTC_SUCCESS
    warn("NVRTC error triggered from:")
    Base.show_backtrace(STDOUT, backtrace())
     bytestring(nvrtcGetErrorString(status)) |> throw
  end
end

export CudaArray, CudaVector, CudaMatrix
include("array.jl")

end # CUDA
