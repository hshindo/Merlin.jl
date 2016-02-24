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

set_backend("cpu")

a = AF.rand(Float32, 1000, 1000)
b = AF.rand(Float32, 1000, 1000)
c = a + b
AF.refcount(b)

AF.refcount(a)
Merlin.Native.logsoft(a.ptr)
b = AF.rand(Float32, 10, 5)
sum(b)
c = to_host(b)
sum(c)
c = Merlin.Native.add!(a, b)
a
Merlin.logsoftmax(a)

function bench()
  a = AF.rand(Float32, 1000, 1000)
  b = AF.rand(Float32, 1000, 1000)
  #a = AF.rand(Float32, 100000)
  #b = AF.rand(Float32, 100000)
  for i = 1:1000
    c = a + b * 1.0f0
    AF.release(c)
    #s = Merlin.Native.vec_sum(a)
    #Merlin.Native.add!(a, b)
    #Merlin.Native.logsoft(a.ptr)
    #Merlin.logsoftmax(a)
  end
  AF.release(a)
  AF.release(b)
end

@time bench()

@time POSTagging.train(path)

Profile.clear()
@profile POSTagging.train(path)
Profile.print()
