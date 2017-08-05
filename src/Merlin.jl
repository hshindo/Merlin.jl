module Merlin

using Base.LinAlg.BLAS

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

abstract type Functor end

include("hdf5.jl")
include("util.jl")
include("var.jl")
include("graph.jl")
include("rand.jl")
#include("native.jl")
include("check.jl")
include("train.jl")

for name in [
    "activation",
    "argmax",
    "blas",
    "cat",
    "crossentropy",
    "dropout",
    "getindex",
    #"glu",
    "gru",
    "gumbel_softmax",
    "linear",
    "lookup",
    "math",
    "reduce",
    "normalize",
    "pairwise",
    #"pooling",
    "reshape",
    "softmax",
    "softmax_crossentropy",
    "window1d",

    "mse",

    "rnn/lstm",
    "rnn/bilstm",
    "cnn/conv1d",
    "cnn/gated_conv1d"
    ]
    include("functions/$(name).jl")
    #isfile(joinpath(dirname(@__FILE__),cudafile)) && include(cudafile)
end

export update!
for name in [
    "adagrad",
    "adam",
    "clipping",
    "sgd"]
    include("optimizers/$(name).jl")
end

#include("caffe/Caffe.jl")

info("# Threads: $(Threads.nthreads())")

end
