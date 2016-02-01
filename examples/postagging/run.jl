push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
workspace()
using ArrayFire
using Merlin
using POSTagging
path = "C:/Users/shindo/Dropbox/tagging"

5 % 3
x = rand(Float32, 10, 5)
xx = AFArray(x)
Merlin.logsoftmax(xx)

Merlin.logsoftmax(x)

m = maximum(xx, 1)
e = exp(xx - m)


l = LogSoftmax()
Merlin.logsoftmax(x)
Merlin.logsoftmax(xx)

function bench()
  xx = rand(Float32, 100, 100)
  x = AFArray(xx)
  for i = 1:1000
    #m = maximum(x, 1)
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
    Merlin.logsoftmax(x)
    #CUBLAS.gemm!('N', 'N', 1.0f0, x, w, 0.0f0, y)
    #y = CUBLAS.gemm('N', 'N', 1.0f0, x, w)
    #i % 10 == 0 && device_gc()
  end
  #release(x)
end

@time bench()

@time POSTagging.train(path)
