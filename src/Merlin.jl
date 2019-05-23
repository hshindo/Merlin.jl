module Merlin

using Base.Threads
# @info "# CPU threads: $(nthreads())"

using Libdl

const CUDA_AVAILABLE = begin
    if Sys.iswindows()
        !isempty(Libdl.find_library("nvcuda"))
    else
        !isempty(Libdl.find_library("libcuda"))
    end
end

if CUDA_AVAILABLE
    include("cuda/CUDA.jl")
    using .CUDA
else
    @info "CUDA not found."
    mutable struct CuArray{T,N} end
    mutable struct CuSubArray{T,N} end
    const CuVector{T} = CuArray{T,1}
    const CuMatrix{T} = CuArray{T,2}
    const CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}
end

using Markdown
import LinearAlgebra.BLAS: scal!, axpy!, gemv, gemv!, gemm, gemm!
export Functor
abstract type Functor end

const UniArray{T,N} = Union{Array{T,N},CuArray{T,N}}
const UniVector{T} = UniArray{T,1}
const UniMatrix{T} = UniArray{T,2}

include("add.jl")
include("var.jl")
include("graph.jl")
include("dataloader.jl")
include("gradient.jl")
include("device.jl")
include("fit.jl")
include("config.jl")

for name in [
    "activation/crelu",
    "activation/elu",
    "activation/leaky_relu",
    "activation/ptanh",
    "activation/relu",
    "activation/selu",
    "activation/sigmoid",
    "activation/swish",
    "activation/tanh",

    "attention/add_attention",

    "cnn/conv1d",
    "cnn/window1d_old",
    # "cnn/conv2d",

    "loss/crossentropy",
    "loss/flip",
    "loss/focalloss",
    "loss/l2",
    "loss/mse",
    "loss/softmax_crossentropy",
    "math/arithmetic",
    "math/broadcast",
    "math/transpose",

    "pooling/avgpooling1d",
    "pooling/maxpooling1d",

    "reduction/average",
    "reduction/argmax",
    "reduction/max",
    "reduction/statistics",
    "reduction/sum",

    "regularization/batchnorm",
    "regularization/dropout",
    "regularization/dropout_dim",
    "regularization/normalize",
    "regularization/weightnorm",
    "regularization/zoneout",

    "rnn/lstm",

    "blas",
    "clip",
    "concat",
    "getindex",
    "linear",
    "lookup",
    "pack",
    "pairwise",
    "repeat",
    "reshape",
    "rnncrf",
    "softmax",
    "sort",
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
include("optimizers/asgd.jl")
include("optimizers/nadam.jl")
include("optimizers/sgd.jl")

include("transformers/identity.jl")
include("transformers/standardizer.jl")

include("datasets/Datasets.jl")
#include("caffe/Caffe.jl")

end
