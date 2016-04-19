ENV["USE_CUDA"] = true
workspace()
using Merlin
using JLD
using Base.LinAlg.BLAS
using Base.Test

@test_approx_eq_eps (1.,) (0.9999,) 1e-3

x = rand(Float32,5)
y = rand(Float32,5)
x+y
check_gradient(Concat(1), x, y)

g1, g2 = approx_gradient(Add(), (x,y))
g1
g2
gg1, gg2 = check_gradient(Add(), x, y)
gg1
gg2
@test_approx_eq_eps

v = Variable(x,nothing)
z = v + y
gradient!(z)
v.grad

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
