module Merlin

abstract Functor
export Functor
abstract Optimizer
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

include("variable.jl")
for name in ["concat","linear","logsoftmax","lookup","relu","reshape","window2d"]
  include("functors/$(name).jl")
end

include("functors/crossentropy.jl") # depends on logsoftmax
include("functors/maxpool2d.jl") # depends on window2d

include("optimizers/adagrad.jl")
include("optimizers/adam.jl")
include("optimizers/sgd.jl")

end
