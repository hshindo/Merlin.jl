module Merlin

using Base.LinAlg.BLAS

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
include("abstractnode.jl")
include("var.jl")
include("sequence.jl")
include("graph.jl")
include("gradient.jl")
include("training.jl")
include("native.jl")
include("serialize.jl")

for name in [
    "activation/relu",
    "activation/sigmoid",
    "activation/tanh",
    "concat",
    "conv",
    "crossentropy",
    "dropout",
    "embedding",
    "gemm",
    "index",
    "linear",
    "math",
    "max",
    "norm",
    "pooling/maxpooling",
    "reshape",
    "softmax",
    "sum",
    "window2"
    ]
    include("functions/$(name).jl")
end

for name in [
    "gru"]
  include("graphs/$(name).jl")
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
