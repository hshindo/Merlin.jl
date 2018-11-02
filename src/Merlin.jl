module Merlin

using Base.Threads
@info "# CPU threads: $(nthreads())"

const GDICT = []

include("cuda/CUDA.jl")
using .CUDA

using Markdown
import LinearAlgebra.BLAS: scal!, axpy!, gemv, gemv!, gemm, gemm!
export gemv, gemm
#export Parametric
#abstract type Parametric end

const UniArray{T,N} = Union{Array{T,N},CuArray{T,N}}
const UniVector{T} = UniArray{T,1}
const UniMatrix{T} = UniArray{T,2}

include("add.jl")
include("var.jl")
include("graph.jl")
include("dataset.jl")
include("check.jl")
include("config.jl")
include("device.jl")

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
    "math/arithmetic",
    "math/broadcast",

    "reduction/max",
    #"reduction/mean",
    #"reduction/sum",

    "rnn/lstm",

    "blas",
    "concat",
    "dropout",
    "getindex",
    "linear",
    "lookup",
    "pack",
    "reshape",
    "softmax",
    "split",
    "window1d"
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
