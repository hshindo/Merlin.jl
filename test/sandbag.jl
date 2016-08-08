workspace()
using Merlin
using Merlin.Caffe
using JuCUDA
using Base.LinAlg.BLAS
using Base.Test
using HDF5
using Compat

a.args[2].args
GraphNode(1)
g = quote
    x = GraphNode(:x)
    x = relu(x)
end
g.args[4].args[2]

g = @graph begin
    T = Float32
    x = GraphNode(:x)
    x = Embedding(T,100,10)(x)
    x
end

g = @graph begin
    T = Float32
    x = Var(:x)
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
