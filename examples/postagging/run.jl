push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
workspace()
using ArrayFire
using Merlin
using POSTagging
path = "C:/Users/shindo/Dropbox/tagging"

x = AFArray(rand(Float32, 10, 2))
f = Window2D(10,4,1,1,0,1)
v = Variable(x)
f(v).value

x = [rand(Float32, 10, 5) for i=1:12]
xx = map(AFArray, x)
cat(1, xx)

macro afcall(ex)
  buffer = []
  local val = $(esc(ex))
  for a in buffer
    release(a)
  end
end

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

ArrayFire.device_gc()
gc()
@time bench()

@time POSTagging.train(path)
