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

typealias DataArray Union{Array,CudaArray}

export argmax
include("util.jl")

include("var.jl")
include("network.jl")
include("trainer.jl")

for name in [
  #"blas",
  "concat",
  "conv",
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
  ]
  include("functors/$(name).jl")
end

for name in [
  Concat,
  Conv,
  CrossEntropy,
  Linear,
  LogSoftmax,
  Lookup,
  Add,ElemAdd,Subtract,ElemSubtract,Mult,ElemMult,
  Max,
  ReLU,
  Reshape,
  Sigmoid,
  Softmax,
  Tanh
  ]
  @eval @compat (f::$name)(args) = forward0(f, args)
  @eval @compat (f::$name)(args...) = f(args)
end

for name in [
    "gru"]
  include("networks/$(name).jl")
end

for name in [
    "adagrad",
    "adam",
    "sgd"]
  include("optimizers/$(name).jl")
end

end
