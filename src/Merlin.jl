module Merlin

using Compat
using Base.LinAlg.BLAS

abstract Optimizer

include("native.jl")

if haskey(ENV, "USE_CUDA")
  using CUDA
  using CUDNN
else
  type CuArray{T,N}
  end
end

include("util.jl")
export argmax
include("var.jl")
export Var, forward, gradient!
include("gradient.jl")
export approx_grad, checkgrad
include("graph.jl")
include("trainer.jl")

macro init0(x, f)
  quote
    $(x).value == nothing && return Var(nothing, $f, [$x], nothing)
  end
end

for name in [
  "activation",
  "concat",
  #"convolution",
  "crossentropy",
  "linear",
  "lookup",
  "math",
  "max",
  "reshape",
  "softmax",
  ]
  include("functors/$(name).jl")
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
