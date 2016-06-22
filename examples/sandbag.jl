workspace()
ENV["USE_CUDA"] = true
using Merlin
using Merlin.Caffe
using CUDA
using Base.LinAlg.BLAS
using Base.Test
using HDF5

path = "C:/Users/shindo/Desktop/nin_imagenet.caffemodel"
g = Caffe.load(path)

show([1,2,3,4,5,6])
