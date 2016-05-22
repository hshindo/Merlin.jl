workspace()
ENV["USE_CUDA"] = true
using Merlin
using CUDA
using JLD
using Base.LinAlg.BLAS
using Base.Test

Native.compile(str, "testname")

str = "#include <stdio.h>\nmain(){puts(\"Hello world\");return 0;}"
cmd = `$(str) | g++ -Wall -O3 -shared -xc -o a.dll -`
run(cmd)

x = CuArray(rand(Float32,5,4,3,2))
xx = Array(x)
f = Convolution(Float32, (3,4), (2,2), (1,1), (0,0))
w = CuArray(f.w.val)
ww = Array(w)

y_cpu = Merlin.convolution(f, xx)
y_cpu = reshape(y_cpu, 12, 24)
y_cpu = reshape(ww, 4, 12) * y_cpu
y_cpu = reshape(y_cpu, 4, 3, 4, 2)
vec(y_cpu)

y_gpu = Merlin.convolution(f, x, w)
y_gpu = Array(y_gpu)
vec(y_gpu)

x = rand(Float32,500,500,10)
f = Convolution(Float32, (3,5), (2,2), (1,1), (0,0))
y = zeros(Float32, 4*499*499*10)
sizes = Cint[500,500,10,2,2,1,1,0,0]

x = rand(Float32,5,4,3)
y = zeros(Float32, 4*3*4*3)
sizes = Cint[5,4,3,2,2,0,0,1,1]
Merlin.conv_test(x, y, sizes)

function bench()
  #a = rand(Float32,100,1000)
  #b = rand(Float32,1000,100)
  for i = 1:1000
    Merlin.conv_test(x, y, sizes)
  end
end
@time bench()
y

x1 = Variable(rand(10,5))
x2 = Variable(rand(10))
x = [x1,x2]
a = map(xx -> xx.val, x)
typeof(a)

x1 = Variable(rand(Float32,7,5),zeros(Float32,7,5))
x2 = Variable(rand(Float32,10,5),zeros(Float32,7,5))
f = Concat(1)
y = (x1,x2) |> f
y.backward!(ones(Float32,17,5))
x2.grad

a = Dict(Float32 => 1)
a[Float32]
typeof(Type{Float32})

x = rand(Float32,10,5)
Merlin.Native.softmax(x, x)
eltype(x)
# w1-w3 are the hidden layer weight matrices, x1 the input vector
function ann(w1, w2, w3, x1)
    x2 = w1 * x1
    x2 = log(1. + exp(x2))   # soft RELU unit
    x3 = w2 * x2
    x3 = log(1. + exp(x3))   # soft RELU unit
    x4 = w3 * x3
    1. / (1. + exp(-x4[1]))  # sigmoid output
end

w1, w2, w3 = randn(10,10), randn(10,10), randn(1,10)
x1 = randn(10)
dann = rdiff(ann, (w1, w2, w3, x1))
dann(w1, w2, w3, x1) # network output + gradient on w1, w2, w3 and x1

softmax_native()
a = CudaArray(Float32,10,5)

function bench()
  for i = 1:10000
    a = CudaArray(Float32,10,5)
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
