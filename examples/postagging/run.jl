push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
ENV["USE_CUDA"] = true
workspace()
using Merlin

function bench()
  a = rand(Float32, 750, 300)
  x = rand(Float32, 300)
  xx = rand(Float32, 300, 30)
  y = zeros(Float32, 750)
  for i = 1:1000
    for j = 1:30
      gemv('N', a, x)
      gemv('N', a, x)
      gemv('N', a, x)
      gemv('N', a, x)
    end
    #gemm('N', 'N', a, xx)
    #Merlin.softmax(r)
    #cat(1,a...)
  end
end

@time begin
  @parallel for i = 1:1000000
    r = rand(Float32, 100, 100)
  end
end

@time bench()

include("token.jl")
include("model_char.jl")
include("train.jl")
path = "C:/Users/hshindo/Dropbox/tagging"

@time train(path)
