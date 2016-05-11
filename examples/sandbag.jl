ENV["USE_CUDA"] = true
workspace()
using Merlin
using CUDA
using JLD
using Base.LinAlg.BLAS
using Base.Test

data_x = [Var(rand(Float32,10,5)) for i=1:100] # input data
data_y = [Var(Int[1,2,3]) for i=1:100] # correct labels

f = Network(
  Linear(Float32,10,7),
  ReLU(),
  Linear(Float32,7,3))
t = Trainer(f, CrossEntropy(), SGD(0.0001))

for epoch = 1:10
  println("epoch: $(epoch)")
  loss = fit(t, data_x, data_y)
  println("loss: $(loss)")
end


f = CrossEntropy()
x1 = Var([1,2,3])
x2 = Var(rand(Float32,10,3))
f([x1,x2])

f = GRU(Float32,10)
f.data_ids

x1 = Var(rand(Float32,10,1))
x2 = Var(rand(Float32,10,1))
f(x1,x2)
y = x1 .- x2
x1.grad = zeros(x1.val)
x2.grad = zeros(x2.val)
backward!(y)

rand(1:5,10)
[1:5]
macro check_grad(a)
  println(a.args[3])
end

@check_grad parse(Int, "5")
0.00016595 < 1e-4
exp(log(0.00020003319) - log(2e-4))
x1 = Var(rand(Float32,5,2))
x2 = Var(rand(Float32,5,2))
checkgrad(Add(),x1,x2)

x = Var(rand(Float32,10,5))
f = Linear(Float32,10,7)
checkgrad(f,x)

@time f(x)
@Merlin.check_grad f(x)

f.w * x .+ f.b
check_gradient(f, x)

a = backward!(y)


f = Conv(Float32,5,(10,2),(1,1),(0,0))
x = Var(rand(Float32,50,10))
y = f(x)

function bench()
  a = [Var(rand(Float32,100,1)) for i=1:30]
  f = Concat(2)
  for i = 1:10000
    b = tuple(a...)
    Merlin.forward(f, b)
  end
end
@time bench()

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
