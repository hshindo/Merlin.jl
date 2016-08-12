workspace()
using Merlin
using Merlin.Caffe
using JuCUDA
using HDF5
using Compat

function bench()
    for i = 1:10000
        #rand(Float32,1000)
        Array(Float32,1000)
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
