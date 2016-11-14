module Merlin

using Base.LinAlg.BLAS
using HDF5

if is_windows()
    const libmerlin = Libdl.dlopen(joinpath(Pkg.dir("Merlin"),"deps/libmerlin.dll"))
elseif is_linux() || is_apple()
    const libmerlin = Libdl.dlopen(joinpath(Pkg.dir("Merlin"),"deps/libmerlin.so"))
else
    throw("Unsupported OS.")
end

abstract Functor
type CuArray{T,N}
end

include("memorypool.jl")
include("session.jl")
include("var.jl")
include("functions/relu.jl")
include("functions/linear.jl")
#include("layers/linear.jl")
#include("layers/relu.jl")
#include("layers/graph.jl")

#include("functors/graph.jl")
#include("functors/linear.jl")
#include("functors/relu.jl")

#=
const USE_CUDA = try
    using JuCUDA
    include("cuda/cudnn/CUDNN.jl")
    using .CUDNN
    true
catch e
    info(e)
    #type CuArray{T,N}; end
    #typealias CuVector{T} CuArray{T,1}
    #typealias CuMatrix{T} CuArray{T,2}
    false
end

typealias UniArray{T,N} Union{Array{T,N},SubArray{T,N},CuArray{T,N}}

include("interop/c/carray.jl")

include("util.jl")
include("var.jl")
include("graph.jl")
include("fit.jl")
include("native.jl")
include("hdf5.jl")
include("check.jl")

for name in [
    #"argmax",
    #"concat",
    #"convolution",
    #"crossentropy",
    #"dropout",
    #"embedding",
    #"exp",
    #"gemm",
    #"getindex",
    #"gru",
    #"linear",
    #"log",
    #"math",
    #"max",
    #"pooling",
    "relu",
    #"reshape",
    #"sigmoid",
    #"softmax",
    #"sum",
    #"tanh",
    #"transpose",
    #"view",
    #"window"
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
=#

end
