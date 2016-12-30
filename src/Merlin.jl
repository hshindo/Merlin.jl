module Merlin

using Base.LinAlg.BLAS

if is_windows()
    const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.dll"))
elseif is_linux() || is_apple()
    const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.so"))
else
    throw("Unsupported OS.")
end

#typealias UniArray{T,N} Union{Array{T,N},CuArray{T,N}}

#include("mkl/MKL.jl")

include("util.jl")
include("var.jl")
include("graph.jl")
include("fit.jl")
include("native.jl")
include("hdf5.jl")
include("gradient.jl")

abstract Functor
for name in [
    "argmax",
    "blas",
    "concat",
    "conv",
    "crossentropy",
    "dropout",
    "getindex",
    "gru",
    "linear",
    "lookup",
    "math",
    "pairwise",
    "pooling",
    "reduce",
    "relu",
    "reshape",
    "sigmoid",
    "softmax",
    "tanh",
    "transpose",
    "view",
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

if !isempty(Libdl.find_library(["nvcuda","libcuda"]))
    include("cuda/CUDA.jl")
    using .CUDA
end

#include("caffe/Caffe.jl")

end
