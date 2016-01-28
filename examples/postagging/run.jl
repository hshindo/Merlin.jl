push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
workspace()
using ArrayFire
using Merlin
using POSTagging
path = "C:/Users/shindo/Dropbox/tagging"

parse(Float32, "1.0")
typeof(float("1.0"))
a = Array(Float32, 10)
push!(a, 1.0)
x = rand(AFArray{Float32}, 10, 5)
y = rand(AFArray{Float32}, 10, 1)
x .* y

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
end

@time bench()

@time POSTagging.train(path)
