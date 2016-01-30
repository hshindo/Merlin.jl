push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
workspace()
using ArrayFire
using Merlin
using POSTagging
path = "C:/Users/shindo/Dropbox/tagging"

x = rand(Float32, 10, 5)
xx = AFArray(x)
m = maximum(xx, 1)
e = exp(xx - m)


l = LogSoftmax()
Merlin.logsoftmax(x)
Merlin.logsoftmax(xx)

function bench()
  x = rand(Float32, 1000, 500)
  xx = AFArray(x)
  for i = 1:1000
    Merlin.logsoftmax(xx)
    #CUBLAS.gemm!('N', 'N', 1.0f0, x, w, 0.0f0, y)
    #y = CUBLAS.gemm('N', 'N', 1.0f0, x, w)
  end
end

@time bench()

@time POSTagging.train(path)
