ENV["USE_CUDA"] = true
workspace()
using Merlin
using JLD
using Base.LinAlg.BLAS
using Base.Test

Ws = [Variable(rand(Float32,100,100)) for i=1:3]
f = GRU(Float32,100,100)
f.inids
f.vars[14].grad
x = rand(Float32,100,1)
h = rand(Float32,100,1)
f(x,h)
check_gradient(f, x, h)

function ttt()
  x = rand(Float64,10,5,2)
  #y = rand(Float32,10,5,2)
  g1 = Merlin.gradient2(Max(3), copy(x))[1]
  g2 = approx_gradient(Max(3), copy(x))
  for k = 1:length(g1)
    d = g1[k] - g2[k]
    if abs(d) >= 1e-4
      println("x...")
      println(k)
      k-50 > 0 && println("k-50: $(x[k-50])")
      k+50 <= length(x) && println("k+50: $(x[k+50])")
      println(x[k])
      println(g1[k])
      println(g2[k])
    end
  end
end
ttt()

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
