module Merlin

using Compat
using Base.LinAlg.BLAS
using JLD

abstract Functor
abstract Optimizer

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

include("variable.jl")
include("graph.jl")
include("sequence.jl")

for name in [
    "logsoftmax",
    "relu",
    "sigmoid",
    "softmax",
    "tanh"]
  include("functors/activation/$(name).jl")
end

for name in [
    "crossentropy"]
  include("functors/loss/$(name).jl")
end

for name in [
    "add",
    "blas",
    "multiply",
    "subtract"]
  include("functors/math/$(name).jl")
end

for name in [
    "concat",
    "linear",
    "lookup",
    "max",
    "reshape",
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
