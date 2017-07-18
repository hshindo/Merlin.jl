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

abstract type Functor end

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
    "dropout",
    "getindex",
    #"glu",
    "gru",
    "linear",
    "lookup",
    #"lstm",
    "math",
    "reduce",
    "normalize",
    "pairwise",
    #"pooling",
    "reshape",
    "softmax",
    "softmax_crossentropy",
    "window",

    "conv1d",
    "mse"
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

include("hdf5.jl")
#include("caffe/Caffe.jl")

info("# threads: $(Threads.nthreads())")

end
