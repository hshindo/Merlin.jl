module Merlin

using Base.LinAlg.BLAS

const libmerlin = Libdl.find_library(["libmerlin"], [joinpath(dirname(@__FILE__),"../deps")])

if !isempty(Libdl.find_library(["nvcuda","libcuda"]))
    include("cuda/CUDA.jl")
    #using .CUDA
end
#typealias UniArray{T,N} Union{Array{T,N},CuArray{T,N}}

#include("mkl/MKL.jl")

include("var.jl")
include("graph.jl")
include("fit.jl")
include("native.jl")
include("hdf5.jl")
include("check.jl")

abstract Functor
for name in [
    "argmax",
    "concat",
    "convolution",
    #"crossentropy",
    #"dropout",
    #"exp",
    #"gemm",
    #"getindex",
    #"gru",
    #"linear",
    #"lookup",
    #"log",
    #"math",
    #"max",
    #"pooling",
    #"reduce",
    #"relu",
    #"reshape",
    #"sigmoid",
    #"softmax",
    #"tanh",
    #"transpose",
    #"view",
    #"window",
    ]
    include("functions/$(name).jl")
    #path = "cuda/functions/$(name).jl"
    #isfile(path) && include(path)
end

export update!
for name in [
    "adagrad",
    "adam",
    "sgd"]
    include("optimizers/$(name).jl")
end

#include("caffe/Caffe.jl")

end
