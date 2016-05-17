module Merlin

using Compat
using Base.LinAlg.BLAS

abstract Functor
abstract Optimizer

include("native.jl")

if haskey(ENV, "USE_CUDA")
  using CUDA
  using CUDNN
else
  type CuArray{T,N}
  end
end

typealias DataArray Union{Array,CuArray}

export argmax
include("util.jl")

include("var.jl")
include("network.jl")
include("trainer.jl")

for name in [
  "activation",
  #"blas",
  "concat",
  "convolution",
  "crossentropy",
  "linear",
  "logsoftmax",
  "lookup",
  "math",
  "max",
  "reshape",
  "softmax",
  ]
  include("functors/$(name).jl")
end

for name in [
  Activation,
  Concat,
  Convolution,
  CrossEntropy,
  Linear,
  LogSoftmax,
  Lookup,
  Add,ElemAdd,Subtract,ElemSubtract,Mult,ElemMult,
  Max,
  Reshape,
  Softmax,
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
