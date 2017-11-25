module Merlin

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

include("initializers/fill.jl")
include("initializers/normal.jl")
include("initializers/orthogonal.jl")
include("initializers/uniform.jl")
include("initializers/xavier.jl")

include("optimizers/adagrad.jl")
include("optimizers/adam.jl")
include("optimizers/sgd.jl")

for name in [
    "activation/crelu",
    "activation/elu",
    "activation/relu",
    "activation/leaky_relu",
    "activation/selu",
    "activation/sigmoid",
    "activation/swish",
    "activation/tanh",

    "attention/add_attention",

    "conv/conv1d",
    "conv/gated_conv1d",

    "loss/crossentropy",
    "loss/l2",
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

include("datasets/Datasets.jl")
#include("caffe/Caffe.jl")

#info("#Threads: $(Threads.nthreads())")

end
