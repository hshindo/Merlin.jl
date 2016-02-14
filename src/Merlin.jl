module Merlin

abstract Functor
abstract Optimizer
export Functor
export Optimizer

using ArrayFire
using Base.LinAlg.BLAS

export Variable, forward!, backward!
export Concat
export CrossEntropy
export Linear
export LogSoftmax
export Lookup
export MaxPool2D
export ReLU
export Reshape
export Sigmoid
export Tanh
export Window2D

export AdaGrad, Adam, SGD, optimize!

import Base: call, getindex, setindex!

# CUDNN
if haskey(ENV, "USE_CUDNN")
  #@windows? (
  #  begin
  #    const libcudnn = Libdl.find_library(["cudnn64_4"])
  #  end : begin
  #    const libcudnn = Libdl.find_library(["libcudnn"])
  #  end)
  #isempty(libcudnn) ? error("CUDNN library cannot be found.") : println("CUDNN is loaded.")

  using CUDNN
  #using CUDArt
end

include("util.jl")
include("variable.jl")
for name in ["concat","crossentropy","linear","logsoftmax","lookup","maxpool","relu","reshape","window2d"]
  include("functors/$(name).jl")
end

include("optimizers/adagrad.jl")
include("optimizers/adam.jl")
include("optimizers/sgd.jl")

end
