workspace()
using Merlin
using Merlin.Caffe
using JuCUDA
using HDF5

T = Float32
ls = [Linear(T,10,7), Linear(T,7,3)]
g = @graph begin
    x = ls[1](:x)
    x = relu(x)
    x = ls[2](x)
    x
end
path = "C:/Users/hshindo/Desktop/hdf5.h5"
h5save(path, g)

x = Var(rand(Float32,10,5), grad=true)
f = Linear(Float32,10,7)
y = f(x)
gradient!(y)

T = Float32
ls = [Linear(T,10,7), Linear(T,7,3)]
g = @graph begin
    x = :x
    x = ls[1](x)
    x = relu(x)
    x = ls[2](x)
    x
end
f = compile(g, :x)
x = Var(rand(Float32,10,5))
y = f(x)

a

path = "C:/Users/hshindo/Desktop/hdf5.h5"
data = [1,"abra",[2,3,4]]

h5save(path, g)
g = h5load(path)
f = compile(g,:x)
x = Var(rand(Float32,10,5))
y = f(x)

embed = Embedding(Float32,100,10)
linear = Linear(Float32,10,7)
g = @graph begin
    x = :x
    x = embed(x)
    x = linear(x)
    x = relu(x)
    x
end
f = compile(g, :x)

h5 = h5read(path, "Merlin")
save_hdf5(path, "g1"=>embed)

load_hdf5(path)["g1"]

path = "C:/Users/shindo/Dropbox/tagging/nyt100.lst"
readdlm(path, ' ', Float32)

f = Embedding(Float32,1000,100)

g = @graph (:x,) begin
    x = :x
    x = relu(x)
    x
end
to_hdf5(g)

v = Var(rand(Float32,10,5))
t = 15
z = [1,3]
save("C:/Users/shindo/Desktop/test.jld", "v", v)

A = rand(10,5)
B = rand(10,5)
A .+= B

x = Param(rand(Float32,4,3))
y = dropout(x, 0.5, true)
gradient!(y)[1].grad
y = window2d(x, (4,2), (1,1), (0,0))
x.data
y.data

x = Param(rand(Float32,5,4,1))
x.data
y = maxpooling(x, (2,2))
gradient!(y)
x.grad

function bench()
    x = Var(rand(Float32,30,50,1))
    for i = 1:10000
        maxpooling(x, (2,2))
    end
end
@time bench()

parse("Merlin.concat")
g = @graph (:x,) begin
    concat(1, :x)
end
dict = Merlin.to_hdf5(g)
save_hdf5(path, dict)
load_hdf5(path)

path = "C:/Users/shindo/Desktop/tokenizer_20.h5"
dict = Dict()
dict["a1"] = [1,2,3,4]
dict["a2"] = 3.4
dict["a3"] = "abracatabra"
save_hdf5(path, dict)
d = load_hdf5(path)

macro aaa(args)
    quote
        $args
    end
end
@aaa (:s,:ss)
function bench()
    x = rand(Float32, 100,100)
    for i = 1:10000
        vecnorm(x, 2)
    end
end

@time bench()

x = Var(rand(10,5))
x[1:1]
y = x[1:5] + x[6:10]
gradient!(y)

x1 = Param(3)
x2 = Param(4)
y = x1+x2
gradient!(y)

x = rand(10,5)
x[(1:5,1:5)]
x = Var(rand(Float32,10,5))
x.data[1:5]
x[1]

y = Param(3) + Param(5)

g = begin
    local l = Linear(Float32,10,5)
    @graph begin
        x = l(:x)
        x = relu(x)
        x
    end
end

f = compile(g)
f(Var(rand(Float32,10,3)))

g(Var(rand(Float32,10,3)))
g.tails

g(rand(10))

g2 = @graph begin
    T = Float32
    x = ExprNode(:x)
    h = ExprNode(:h)
    x = @lazy Linear(T,10,3)(x)
    h = @lazy Linear(T,10,3)(x)
    concat(1,x,h)
end
f = eval(g2)
f(Var(rand(Float32,10,4)), Var(rand(Float32,10,4)))

x = rand(Float32,10)
eval(g)

g.args[2].args

g = @graph begin
    T = Float32
    x = GraphNode(:x)
    x = reshape(x,1,length(x))
    x = Embedding(T,100,10)(x)
    x = Conv(T, (10,7), (1,70), paddims=(0,3))(x)
    x = reshape(x, size(x,2), size(x,3))
    x = transpose(x)
    x = relu(x)
    x = Linear(T,70,4)(x)
    x
end
g

x = Var(reshape(chars,1,length(chars)))
x = m.embed(x)
x = m.conv(x)
x = reshape(x, size(x,2), size(x,3))
x = transpose(x)
x = relu(x)
x = m.linear(x)

nprocs()
path = "C:/Users/hshindo/Desktop/nin_imagenet.caffemodel"
g = Caffe.load(path)
g.nodes

function bench()
    for i = 1:10000
        @simd for j = 1:10000
            a = rand(Float32)
        end
        #rand(Float32,100,100)
        #a * b
    end
end
@time bench()
