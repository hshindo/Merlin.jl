workspace()
ENV["USE_CUDA"] = true
using Merlin
using JuCUDA
#using Merlin.Caffe
#using JuCUDA
using HDF5
using JLD

T = Float32
x = CuArray(T,5,4,3,2)
y = relu(x)

path = "C:/Users/shindo/Desktop/hdf5.h5"
save(path, "t", f)

gru = GRU(Float32,100)

T = Float32
ls = [Linear(T,10,7), Linear(T,7,3)]
g = @graph begin
    x = :x
    h = :h
    x = ls[1](x)
    x = relu(x)
    x = ls[2](x)
    x
end
g(constant(rand(T,10,3)), constant(rand(T,10,3)))
h5save(path, g)

path = "C:/Users/hshindo/Desktop/hdf5.h5"
h5save(path, g)

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
