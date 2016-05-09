ENV["USE_CUDA"] = true
workspace()
using Merlin
using CUDA
using JLD
using Base.LinAlg.BLAS
using Base.Test

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
