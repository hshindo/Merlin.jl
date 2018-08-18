module Merlin

using Base.Threads
@info "# CPU threads: $(nthreads())"

include("cuda/CUDA.jl")
using .CUDA

using Markdown
# const TEMP_CUDA = LibCUDA.AtomicMalloc()
#export Functor
#abstract type Functor end

const UniArray{T,N} = Union{Array{T,N},CuArray{T,N}}
const UniVector{T} = UniArray{T,1}
const UniMatrix{T} = UniArray{T,2}

include("config.jl")
include("add.jl")
include("var.jl")
include("graph.jl")
include("test.jl")
#include("iterators.jl")

for name in [
    "activation/crelu",
    "activation/elu",
    "activation/leaky_relu",
    "activation/relu",
    "activation/selu",
    "activation/sigmoid",
    "activation/swish",
    "activation/tanh",

    "cnn/conv1d",
    # "cnn/conv2d",

    "loss/crossentropy",
    "loss/l2",
    "loss/mse",
    "loss/softmax_crossentropy",

    #"reduction/argmax",
    "reduction/max",
    #"reduction/mean",
    #"reduction/sum",

    "rnn/lstm",
    "rnn/transpose_batch",

    "blas",
    "concat",
    "dropout",
    "getindex",
    "linear",
    "lookup",
    "math",
    "pack",
    "reshape",
    "softmax",
    "split"
    ]
    include("functions/$name.jl")
end

include("initializers/fill.jl")
include("initializers/normal.jl")
include("initializers/orthogonal.jl")
include("initializers/orthonormal.jl")
include("initializers/uniform.jl")
include("initializers/xavier.jl")

include("optimizers/adagrad.jl")
include("optimizers/adam.jl")
include("optimizers/sgd.jl")

#include("datasets/Datasets.jl")
#include("caffe/Caffe.jl")

end
