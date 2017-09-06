module Merlin

using Base.LinAlg.BLAS

if is_windows()
    const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.dll"))
elseif is_linux() || is_apple()
    const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.so"))
else
    throw("Unsupported OS.")
end

mutable struct Config
    train::Bool
    debug::Bool
end
const config = Config(true, false)

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
include("abstractvar.jl")
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
    "dropout",
    "getindex",
    "linear",
    "logsoftmax",
    "lookup",
    "math",
    "reshape",
    "softmax",
    "standardize",
    "window1d",

    "cnn/conv1d",
    "cnn/gated_conv1d",
    "loss/crossentropy",
    "loss/softmax_crossentropy",
    "reduce/max",
    "reduce/sum",
    "rnn/lstm",
    "rnn/bilstm"
    ]
    include("functions/$(name).jl")
    #isfile(joinpath(dirname(@__FILE__),cudafile)) && include(cudafile)
end

export update!
for name in [
    "adagrad",
    "adam",
    "sgd"]
    include("optimizers/$(name).jl")
end

#include("caffe/Caffe.jl")

info("# Threads: $(Threads.nthreads())")

end
