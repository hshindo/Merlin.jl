push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
push!(LOAD_PATH, "C:/Program Files/ArrayFire/v3/lib")
ENV["USE_CUDA"] = true
workspace()
using IterativeSolvers
using ArrayFire
using Merlin
using POSTagging
path = "C:/Users/shindo/Dropbox/tagging"

Libdl.find_library(["cudnn64_4"])
ArrayFire.device_info()
AFArray([1,2,3])

function bench()
  xx = [rand(Float32, 100, 1) for i=1:100]
  x = map(AFArray, xx)
  for i = 1:1000
    #m = maximum(x, 1)
    #y = Merlin.logsoftmax(x)
    #mm = x - m
    #e = exp(mm)
    #z = sum(e, 1)
    #logz = log(z)
    #y = mm - logz
    #release(m)
    #release(mm)
    #release(e)
    #release(z)
    #release(logz)
    #release(y)
    y = Merlin.concat(1, x)
    release(y)
    #CUBLAS.gemm!('N', 'N', 1.0f0, x, w, 0.0f0, y)
    #y = CUBLAS.gemm('N', 'N', 1.0f0, x, w)
    #i % 10 == 0 && device_gc()
  end
  map(release, x)
  #release(x)
end

@time bench()

@time POSTagging.train(path)

Profile.clear()
@profile POSTagging.train(path)
Profile.print()
