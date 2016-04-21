module Merlin

using Compat
using Base.LinAlg.BLAS
using JLD

abstract Functor
abstract Optimizer

include("native.jl")
include("cuda/cudaarray.jl")

if haskey(ENV, "USE_CUDA")
  using CUDA
  using CUDA.RT
  using CUDNN
end

typealias Data{T,N} Union{Array{T,N},CudaArray{T,N}}

include("variable.jl")
include("graph.jl")
include("sequence.jl")

for name in ["add",
             "blas",
             "concat",
             "crossentropy",
             "linear",
             "logsoftmax",
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
