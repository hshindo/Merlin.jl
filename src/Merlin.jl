module Merlin

using Base.Threads
info("# CPU threads: $(nthreads())")

include("cuda/CUDA.jl")
using .CUDA

# const TEMP_CUDA = LibCUDA.AtomicMalloc()
export Functor
abstract type Functor end

const UniArray{T,N} = Union{Array{T,N},CuArray{T,N}}
const UniVector{T} = UniArray{T,1}
const UniMatrix{T} = UniArray{T,2}

include("config.jl")
include("add.jl")
include("var.jl")
include("graph.jl")
include("test.jl")
include("initializer.jl")
include("optimizer.jl")
include("iterators.jl")

for name in [
    "activation/crelu",
    "activation/elu",
    "activation/leaky_relu",
    "activation/relu",
    "activation/selu",
    "activation/sigmoid",
    "activation/swish",
    "activation/tanh",

    "loss/crossentropy",
    "loss/l2",
    "loss/mse",
    "loss/softmax_crossentropy",

    "reduction/argmax",
    "reduction/max",
    "reduction/mean",
    "reduction/sum",

    "blas",
    "concat",
    "conv",
    "dropout",
    "getindex",
    "linear",
    "lookup",
    "math",
    "pack",
    "reshape",
    "rnn",
    "softmax",
    "split"
    ]
    include("functions/$name.jl")
end

include("datasets/Datasets.jl")
#include("caffe/Caffe.jl")

end
