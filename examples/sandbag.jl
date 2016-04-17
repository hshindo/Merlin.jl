ENV["USE_CUDA"] = true
workspace()
using Merlin
using JLD
using Base.LinAlg.BLAS

x1 = Variable(rand(Float32,5),nothing)
x2 = rand(Float32,5)
v = x1 + x2
x1.value

a = rand(Float32,10)
b = 0 + a
b[1] = 1.0f0
b[1]

3 + x1
x1 + 5

gru = GRU(Float32,50,50)

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
