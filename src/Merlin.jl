module Merlin

const VERSION = v"0.1.0"

using Base.LinAlg.BLAS
using JLD2

if is_windows()
    const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.dll"))
elseif is_linux() || is_apple()
    const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.so"))
else
    throw("Unsupported OS.")
end

#=
if haskey(ENV,"USE_CUDA") && ENV["USE_CUDA"]
    using CUJulia
    include("cuda/cudnn/CUDNN.jl")
    using .CUDNN
else
    type CuArray{T,N}; end
    CuVector{T} = CuArray{T,1}
    CuMatrix{T} = CuArray{T,2}
end
=#

#include("hdf5.jl")
include("graph.jl")
include("var.jl")
#include("native.jl")
include("check.jl")

for name in [
    "const",
    "normal",
    "orthogonal",
    "uniform",
    "xavier"
    ]
    include("initializers/$(name).jl")
end

for name in [
    "activation/crelu",
    "activation/elu",
    "activation/relu",
    "activation/leaky_relu",
    "activation/selu",
    "activation/sigmoid",
    "activation/tanh",

    "attention/add_attention",

    "conv/conv1d",
    "conv/gated_conv1d",

    "loss/crossentropy",
    "loss/mse",
    "loss/softmax_crossentropy",

    "random/dropout",

    "reduction/max",
    "reduction/sum",

    "recurrent/lstm",
    "recurrent/bilstm",

    "argmax",
    "blas",
    "concat",
    "embeddings",
    "getindex",
    "linear",
    "logsoftmax",
    "math",
    "pairwise",
    "reshape",
    "resize",
    "softmax",
    "split",
    "standardize"
    ]
    include("functions/$(name).jl")
    #isfile(joinpath(dirname(@__FILE__),cudafile)) && include(cudafile)
end

for name in [
    "l2"]
    include("regularizers/$(name).jl")
end

export update!
for name in [
    "adagrad",
    "adam",
    "sgd"]
    include("optimizers/$(name).jl")
end

#include("caffe/Caffe.jl")

info("#Threads: $(Threads.nthreads())")

end
