push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
ENV["USE_CUDA"] = true
workspace()
using IterativeSolvers
using ArrayFire
using Merlin
using CUDNN
using POSTagging
path = "C:/Users/hshindo/Dropbox/tagging"

setbackend("cpu")

x = rand(AFArray, Float32, 5, 1)
w = rand(AFArray, Float32, 5, 1)

unwrap(x, 10, 2, 1, 1, 0, 0)

range2(AFArray, Float32, (5,3), 3)
xx = [rand(Float32, 100, 1) |> AFArray for i=1:100]
cat(1, xx)

function bench()
  xx = [rand(Float32, 100, 1) for i=1:100]
  x = map(AFArray, xx)
  for i = 1:1000
    y = Merlin.concat(1, x)
    release(y)
    #CUBLAS.gemm!('N', 'N', 1.0f0, x, w, 0.0f0, y)
    #y = CUBLAS.gemm('N', 'N', 1.0f0, x, w)
    #i % 10 == 0 && device_gc()
  end
  map(release, x)
  #release(x)
end
ArrayFire.af_ptrs
ArrayFire.reset()
@time bench()

@time POSTagging.train(path)

Profile.clear()
@profile POSTagging.train(path)
Profile.print()
