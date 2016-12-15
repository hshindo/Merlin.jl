module Merlin

using Base.LinAlg.BLAS
using HDF5

if is_windows()
    const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.dll"))
elseif is_linux() || is_apple()
    const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.so"))
else
    throw("Unsupported OS.")
end

include("mkl/MKL.jl")

if Pkg.installed("CUDA") != nothing
    using CUDA
    using CUDA.CUDNN
else
    type CuArray{T,N}
    end
    typealias CuVector{T} CuArray{T,1}
    typealias CuMatrix{T} CuArray{T,2}
    macro nvrtc(ex...)
    end
end

typealias XPUArray{T,N} Union{Array{T,N},CuArray{T,N}}

#include("interop/c/carray.jl")

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
    "crossentropy",
    #"dropout",
    #"exp",
    "gemm",
    #"getindex",
    #"gru",
    "linear",
    "lookup",
    #"log",
    "math",
    "max",
    #"pooling",
    #"reduce",
    "relu",
    #"reshape",
    "sigmoid",
    "softmax",
    "tanh",
    #"transpose",
    #"view",
    "window",
    ]
    include("functions/$(name).jl")
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
