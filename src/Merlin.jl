module Merlin

using Base.LinAlg.BLAS
using HDF5

abstract Functor

if is_windows()
    const libmerlin = Libdl.dlopen(joinpath(Pkg.dir("Merlin"),"deps","libmerlin.dll"))
else
    const libmerlin = Libdl.dlopen(joinpath(Pkg.dir("Merlin"),"deps","libmerlin.so"))
end

export cuda_available
cuda_available() = isdir(joinpath(Pkg.dir(),"JuCUDA")) && isdir(joinpath(Pkg.dir(),"JuCUDNN"))

if cuda_available()
    using JuCUDA, JuCUDNN
else
    info("JuCUDA or JuCUDNN is not loaded.")
    type CuArray{T,N}; end
    typealias CuVector{T} CuArray{T,1}
    typealias CuMatrix{T} CuArray{T,2}
end

typealias UniArray{T,N} Union{Array{T,N},CuArray{T,N}}

include("interop/c/carray.jl")

include("util.jl")
include("var.jl")
include("graph.jl")
include("gradient.jl")
include("training.jl")
include("native.jl")
include("hdf5.jl")

for name in [
    "add",
    "axsum",
    "concat",
    "conv",
    "crossentropy",
    "dropout",
    "embedding",
    "exp",
    "gemm",
    "getindex",
    "gru",
    "linear",
    "log",
    "logsoftmax",
    "max",
    "maxpooling",
    "multiply",
    "norm",
    "relu",
    "reshape",
    "sigmoid",
    "softmax",
    "sum",
    "tanh",
    "transpose",
    "view",
    "window2"
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
