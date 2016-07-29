workspace()
using Merlin
using Merlin.Caffe
using JuCUDA
using Base.LinAlg.BLAS
using Base.Test
using HDF5
using Compat



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
