module Merlin

using Compat
using Base.LinAlg.BLAS
using JLD

abstract Functor
abstract Optimizer

export CudaArray

include("native.jl")

if haskey(ENV, "USE_CUDA")
  #push!(LOAD_PATH, joinpath(dirname(@__FILE__), "cuda"))
  using CUDA
else
  type CudaArray{T,N} # Dummy
  end
end

#include("util.jl")
include("variable.jl")
include("graph.jl")
include("sequence.jl")

for name in ["add",
             "blas",
             "concat",
             "crossentropy",
             "linear",
             "lookup",
             "max",
             "multiply",
             "relu",
             "reshape",
             "sigmoid",
             "softmax",
             "subtract",
             "tanh",
             "window2d"]
  include("functors/$(name).jl")
end

for name in ["gru"]
  include("graphs/$(name).jl")
end

for name in ["adagrad",
             "adam",
             "sgd"]
  include("optimizers/$(name).jl")
end

end
