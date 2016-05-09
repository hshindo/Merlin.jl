module Merlin

using Compat
using Base.LinAlg.BLAS

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

export argmax
include("util.jl")

include("variable.jl")
include("graph.jl")
include("sequence.jl")
include("training.jl")

#=
for name in [
  "concat",
  "crossentropy",
  "linear"]
  include("functors2/$(name).jl")
end
=#


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

#=
for name in [
    "gru"]
  include("graphs/$(name).jl")
end
=#

for name in [
    "adagrad",
    "adam",
    "sgd"]
  include("optimizers/$(name).jl")
end

end
