module Merlin

using Base.Threads
info("# CPU threads: $(nthreads())")
using LibCUDA

export Functor
abstract type Functor end

include("config.jl")
include("add.jl")
include("var.jl")
include("graph.jl")
include("test.jl")
include("initializer.jl")
include("optimizer.jl")
include("iterators.jl")

for name in [
    "activation",
    "argmax",
    "blas",
    "concat",
    "conv",
    "dropout",
    "getindex",
    "linear",
    "lookup",
    "loss",
    "math",
    "pad",
    "reduce",
    "reshape",
    "rnn",
    "softmax",
    "split",
    "standardize",
    "transpose_batch",
    "window1d"
    ]
    include("functions/$name.jl")
    f = joinpath(@__DIR__, "cuda/functions/$name.jl")
    isfile(f) && include(f)
end

include("datasets/Datasets.jl")
#include("caffe/Caffe.jl")

end
