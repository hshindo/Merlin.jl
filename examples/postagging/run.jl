github_path = "C:/Users/shindo/Documents/GitHub"
push!(LOAD_PATH, github_path)
push!(LOAD_PATH, dirname(@__FILE__))
workspace()
using Merlin
using POSTagging
path = "C:/Users/shindo/Dropbox/tagging"

using CUDArt
using Merlin.CUDNN
using CUBLAS

a = rand(10)
Array[a]
function bench()
  r = rand(Float32, 1000, 1000) - 0.5
  x = CudaArray(r)
  w = CudaArray(r)
  y = CudaArray(r)
  for i = 1:1000
    #CUBLAS.gemm!('N', 'N', 1.0f0, x, w, 0.0f0, y)
    y = CUBLAS.gemm('N', 'N', 1.0f0, x, w)
  end

  #y = CudaArray(r)
  #xdesc = CUDNN.create_tensor_descriptor(x)
  #ydesc = CUDNN.create_tensor_descriptor(y)
  #for i = 1:4000
  #  p = CUDArt.malloc(Float32, 1000*1000)
  #  CUDArt.free(p)
    #CUDNN.activation_forward(CUDNN.ACTIVATION_RELU, x, y)
    #CUDNN.activation_forward2(CUDNN.ACTIVATION_RELU, x, xdesc, y, ydesc)
  #end
end

@time bench()

@time POSTagging.train(path)
