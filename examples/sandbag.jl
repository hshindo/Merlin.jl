workspace()
ENV["USE_CUDA"] = true
using Merlin
using Merlin.Caffe
using CUDA
using JLD
using Base.LinAlg.BLAS
using Base.Test
using HDF5

function bench()
  x = rand(Float32,100,100)
  for i = 1:1000
    #Merlin.softmax(x)
    Merlin.softmax_native(x)
  end
end
@time bench()

gru = GRU(Float32,100)
x = Var(rand(Float32,100,1))
h = Var(rand(Float32,100,1))
gru(:x=>x, :h=>h)

T = Float32
x1 = Var(rand(T,10,5))
x3 = Var(rand(T,10,1))
x1 + x3

v = Var(rand(10))
relu(v)
v = Var(:x)
relu(v)

function bench()
  x = rand(Float32,10000)
  y = similar(x)
  for i = 1:1000
    normalexp!(x, y)
    #fastexp!(x, y)
  end
end

@time bench()

function times{T}(xs::Vector{Matrix{T}})
  Var(xs[1] * xs[2]), nothing
end
function times{T}(x1::Matrix{T}, x2::Matrix{T})
  Var(x1 * x2), nothing
end

function bench()
  vars = [Merlin.Var2(rand(Float32,5,5)) for i=1:100]
  for i = 1:100000
    xs = @fastmap (v -> v.value) Array{Float32,2} vars
    #xs = Array(Array{Float32,2}, length(vars))
    #for j = 1:length(vars)
    #  xs[j] = vars[j].value
    #end
    #xs = map(v -> v.value, vars)
  end
  #x1 = Var(rand(Float32,20,10))
  #x2 = Var(rand(Float32,10,10))
  #x3 = Var(rand(Float32,5,10))
  #=
  for i = 1:100000
    args = [x1,x2]
    any(v -> typeof(v.value) <: Symbol, args) && continue
    a = []
    for i = 1:2
      push!(a, args[i].value)
    end
    times(a...)
    #Var(x1.value * x2.value)
    #x1 + x2 + x3
    #forward(Merlin.Plus([1,1,1]), [x1,x2,x3])
  end
  =#
end
@time bench()

x = Var(rand(Float32,3))
macroexpand(:(@checkgrad relu x)) |> println
@checkgrad relu x x

macroexpand(@checkgrad relu x)

y = relu(x)
gradient!(y)

w = Var(rand(Float32,2,2,3,4))
x = Var(rand(Float32,5,4,3,2))
y = conv(w, x, stride=(1,1), pad=(0,0))

ww = Var(CuArray(w.value))
xx = Var(CuArray(x.value))
y = conv(w, x, stride=(1,1), pad=(0,0))

x = param(rand(Float32,10,5))
y = sigmoid(x)
gradient!(y)
x.grad

Merlin.empty(rand(10))
h5write("C:/Users/shindo/Desktop/test.h5", "A", rand(Float32,10))

type AW
  dim::Int
  data::Vector{Int}
end

a = AW(1, [2,3,4])
JLD.save("C:/Users/shindo/Desktop/test2.jld", "x", a)

d = Dict("a"=>[1,2,3,4], "b"=>2)
h5writeattr("C:/Users/shindo/Desktop/test.h5", "nn", d)
x = h5readattr("C:/Users/shindo/Desktop/test.h5", "nn")

x = Var(rand(10,5))
y = relu(x)
x = Var(:x)
y = relu(x)
checkgrad(()->relu(x), x)

f = Window2D(2,2,1,1,0,0)
x = Var(rand(Float32,3,3))
f(x)

x = Var(rand(Float32,10,5))
f = Linear(Float32,10,7)
y = f(x)

f = Lookup(Float32,100,10) # 100-length vector, 10k vocabulary
x = rand(1:10,3,2)
y = f(x)

gru = GRU(Float32,100)
x = param(rand(Float32,100,1))
h = param(rand(Float32,100,1))
y = gru(:x=>x, :h=>h)

y = gru(:x => , :h => Var(rand(Float32,100)))
y.value

x = Var(rand(Float32,10,5))
f = Linear(Float32,10,7)
f(x)

x = [param(rand(Float32,100,100)) for i=1:10]

function bench()
  r1 = [1,2,3,4]
  #r1 = rand(100,100)
  #r2 = rand(100,100)
  for i = 1:10000
    a1(r1...)
    #a2(r1)
  end
end

@time bench()

np = Caffe.load("C:/Users/hshindo/Desktop/VGG_ILSVRC_19_layers.caffemodel")
p = np["conv5_3"].convolution_param

x1 = Var(rand(10,5))
y = relu(x1)
y.value

x2 = rand(5,7)
y = zeros(10,7)
gemm!('N', 'N', 1., x1, x2, 0., y)

Var(rand(Float32,5,4))

x = Var(rand(Float32,5,4,3,2))
w = Var(rand(Float32,2,2,3,4)) # 2-d convolution
f = Conv(w, Var(), (1,1), (0,0))
y = f(x)

xx = CuArray(x.value)
ww = CuArray(w.value)
yy = conv(f, xx, ww)
z = Array(yy)
z - y

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

dir = joinpath(dirname(@__FILE__), "..", "lib")
libname = "softmax_float_10_5.dll"
libpath = joinpath(dir, libname)
h = Merlin.Native.softmax_float_10_5
Libdl.dlclose(Merlin.Native.)

x = rand(Float32,10,10)
softmax(x)

dir = joinpath(dirname(@__FILE__), "..", "deps")
const HANDLE = Libdl.dlsym(Merlin.Native.library, :softmax_fw_f32)

function bench()
  x1 = rand(500,100,3)
  x2 = rand(100,100)
  for i = 1:1000
    for j = 1:3
      #a = pointer_to_array(pointer(x1, (j-1)*50000), (500, 100))
      gemm('N', 'N', slice(x1,:,:,j), x2)
    end
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
