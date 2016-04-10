push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../.."))
push!(LOAD_PATH, dirname(@__FILE__))
ENV["USE_CUDA"] = true
workspace()
using Merlin
using JLD
using Base.LinAlg.BLAS

a = rand(10,10)
Array[a]
convert(Matrix{Float32}, a)

v1 = Variable(rand(Float32,10,10))
v2 = Variable(rand(Float32,10,10))

v1 + v2

function bench()
  #A = rand(Float32,500,500)
  #B = rand(Float32,500,500)
  #C = zeros(Float32,500,30)
  for i = 1:10000
    a = Array(Float32,1000)
    #axpy!(-1.0f0, C, A*B)
    #D = A * B
    #broadcast!(+, B, B, C)
    #D = B + C
    #for ii = 1:10
    #  v = Variable()
    #end
  end
end

@time bench()

path = "C:/temp/"
A = reshape(1:120, 15, 8)
A = AAA(A)
save("$(path)/A.jld", "A", A)
v = load("$(path)/A.jld", "A")
