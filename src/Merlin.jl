module Merlin

using Compat
using Base.LinAlg.BLAS

abstract Functor
abstract Optimizer
export update!

include("native.jl")
#include("cuda/cudaarray.jl")

if haskey(ENV, "USE_CUDA")
  #using CUDA
  #using CUDA.RT
  #using CUDNN
end
type CudaArray{T,N}
end

typealias Data{T,N} Union{Array{T,N},CudaArray{T,N}}

export argmax
include("util.jl")

include("variable.jl")
include("graph.jl")
include("sequence.jl")
include("training.jl")

for name in [
  "blas",
  "concat",
  "crossentropy",
  "linear",
  "logsoftmax",
  "lookup",
  "math",
  "max",
  "relu",
  "reshape",
  "sigmoid",
  "softmax",
  "tanh",
  "window2d"]
  include("functors/$(name).jl")
end

for name in [
    "gru"]
  include("graphs/$(name).jl")
end

for name in [
    "adagrad",
    "adam",
    "sgd"]
  include("optimizers/$(name).jl")
end

end
