workspace()
using Merlin
using Merlin.Caffe
using JuCUDA
using Base.LinAlg.BLAS
using Base.Test
using HDF5
using Compat

x = Var(rand(Float32,5,4,3,2))
f = Conv(Float32, (2,2), (3,4), stride=(1,1), paddims=(0,0))
y = f(x)

function bench()
    dict = Dict(:1=>1,:2=>2,:3=>3,:4=>4,:5=>5,:6=>6,:7=>7,:8=>8,:9=>9,:10=>10)
    for i = 1:10000
        for k = 1:10
            dict[k]
        end
    end
end

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
